#!/usr/bin/env node
/*
 * build_deck.js - builds the BedSynth AI final presentation (BedSynth_AI.pptx) from the
 * project's saved results. Every number on a slide is read or computed from the input files;
 * nothing is typed by hand, so re-running this script after a new pipeline run refreshes
 * the whole deck.
 *
 *   node build_deck.js                         # defaults: inputs under the project root
 *   node build_deck.js --reports-dir X --models-dir Y --out Z.pptx
 *
 * Inputs (all required; the build stops with an error naming the missing file or field):
 *   <reports>/results/model_comparison.json, the per-model and reference result JSONs it lists,
 *   <reports>/results/pytest_junit.xml, <reports>/api_demo/summary.json (+ the evidence files
 *   of the steps used below), <models>/registry.json, <models>/<table>/<model>/model_card.json,
 *   <processed>/data_quality_report.json, <processed>/schema_catalog.json,
 *   <processed>/ward_daily_clean.csv (seasonal occupancy chart).
 * Optional: <reports>/results/pytest_junit_slow.xml (slow test suite, written by
 *   scripts/05_api_demo.py --run-slow-tests), <reports>/experiments/ward_daily_band_groups_ab.json
 *   (A/B test of the ward_daily gc_tuned spec with its re-scores on further evaluation seeds; the
 *   notes of slides 13 and 19 quote it when it describes the current ward_daily default, and keep
 *   a single-seed caveat otherwise) and docs/presentation/assets/ui_screenshot.png (screenshot of
 *   the /ui demo page).
 *
 * Speaker notes are timed for the 20-minute slot of docs/TEAM_PLAN.md (section 5): about
 * 17 minutes of slides plus about 3 minutes of live demo after the testing slide.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const PptxGenJS = require('pptxgenjs');
const JSZip = require('jszip');

const SCRIPT_DIR = __dirname;
// Display names of the four speakers, the placeholders docs/TEAM_PLAN.md uses. Put the team's names
// here (short names fit the speaker badge best): they replace "Member A" ... "Member D" on the slides,
// in the speaker notes and in the section names; the badges keep the letters.
const MEMBERS = { A: 'Member A', B: 'Member B', C: 'Member C', D: 'Member D' };
const named = (text) => String(text).replace(/\bMember ([A-D])\b/g, (m, k) => MEMBERS[k] || m);
const TABLES = ['ward_daily', 'patient_stays'];
const ABLATION = ['gc_baseline', 'gc_segmented', 'gc_tuned'];
const PREFERRED_ORDER = ['gc_baseline', 'gc_segmented', 'gc_tuned', 'ctgan', 'tvae'];
const DUPLICATE_KINDS = ['exact_duplicate', 'key_duplicate']; // detected kinds counted as duplicate_row
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

// ------------------------------------------------------------------------------------------
// Command line
// ------------------------------------------------------------------------------------------
const USAGE = `Usage: node build_deck.js [options]

  --root DIR            project root (default: two levels up from this script)
  --reports-dir DIR     reports directory (default: <root>/reports)
  --models-dir DIR      models directory (default: <root>/models)
  --processed-dir DIR   processed-data directory (default: <root>/data/processed)
  --out FILE            output deck (default: BedSynth_AI.pptx next to this script)
  --ui-screenshot FILE  screenshot of the /ui page (default: assets/ui_screenshot.png next to
                        this script, used when present)
  -h, --help            show this help

Relative paths are resolved against the current directory.`;

class InputError extends Error {}

function fail(message) {
  throw new InputError(message);
}

function parseArgs(argv) {
  const names = {
    '--root': 'root',
    '--reports-dir': 'reportsDir',
    '--models-dir': 'modelsDir',
    '--processed-dir': 'processedDir',
    '--out': 'out',
    '--ui-screenshot': 'uiScreenshot',
  };
  const opts = {};
  for (let i = 0; i < argv.length; i += 1) {
    let arg = argv[i];
    if (arg === '-h' || arg === '--help') {
      console.log(USAGE);
      process.exit(0);
    }
    let value;
    const eq = arg.indexOf('=');
    if (arg.startsWith('--') && eq > 0) {
      value = arg.slice(eq + 1);
      arg = arg.slice(0, eq);
    }
    const key = names[arg];
    if (!key) fail(`unknown option "${arg}"\n\n${USAGE}`);
    if (value === undefined) {
      value = argv[i + 1];
      i += 1;
      if (value === undefined || value.startsWith('--')) fail(`option ${arg} needs a value`);
    }
    opts[key] = value;
  }
  const root = path.resolve(opts.root || path.join(SCRIPT_DIR, '..', '..'));
  return {
    root,
    reportsDir: path.resolve(opts.reportsDir || path.join(root, 'reports')),
    modelsDir: path.resolve(opts.modelsDir || path.join(root, 'models')),
    processedDir: path.resolve(opts.processedDir || path.join(root, 'data', 'processed')),
    out: path.resolve(opts.out || path.join(SCRIPT_DIR, 'BedSynth_AI.pptx')),
    uiScreenshot: path.resolve(opts.uiScreenshot || path.join(SCRIPT_DIR, 'assets', 'ui_screenshot.png')),
    uiScreenshotRequired: Boolean(opts.uiScreenshot),
  };
}

// ------------------------------------------------------------------------------------------
// Input loading (fails loudly)
// ------------------------------------------------------------------------------------------
function mustExist(file, what) {
  if (!fs.existsSync(file)) fail(`missing input (${what}): ${file}`);
  return file;
}

function readText(file, what) {
  return fs.readFileSync(mustExist(file, what), 'utf8');
}

function readJson(file, what) {
  const text = readText(file, what);
  try {
    return JSON.parse(text);
  } catch (first) {
    try {
      const data = JSON.parse(text.replace(/(?<=[:[,]\s*)-?(?:NaN|Infinity)(?=\s*[,\]}])/g, 'null'));
      console.warn(`warning: ${what} contains NaN/Infinity values; they were read as null`);
      return data;
    } catch (second) {
      return fail(`cannot parse ${what} (${file}): ${first.message}`);
    }
  }
}

/** Required field: pick(obj, 'a.b.c', 'what') - fails with the field path when it is absent. */
function pick(obj, keyPath, what) {
  let cur = obj;
  for (const key of String(keyPath).split('.')) {
    if (cur === null || cur === undefined || typeof cur !== 'object' || !(key in cur)) {
      fail(`${what}: missing field "${keyPath}"`);
    }
    cur = cur[key];
  }
  if (cur === null || cur === undefined) fail(`${what}: field "${keyPath}" is empty`);
  return cur;
}

/** Optional field with a fallback. */
function opt(obj, keyPath, fallback = null) {
  let cur = obj;
  for (const key of String(keyPath).split('.')) {
    if (cur === null || cur === undefined || typeof cur !== 'object' || !(key in cur)) return fallback;
    cur = cur[key];
  }
  return cur === null || cur === undefined ? fallback : cur;
}

function num(value, what) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${what}: expected a number, got ${JSON.stringify(value)}`);
  return value;
}

function pngSize(file) {
  const buf = fs.readFileSync(file);
  if (buf.length < 24 || buf.readUInt32BE(0) !== 0x89504e47) fail(`not a PNG image: ${file}`);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function parseAttrs(text) {
  const out = {};
  for (const m of text.matchAll(/([\w:.-]+)\s*=\s*"([^"]*)"/g)) out[m[1]] = m[2];
  return out;
}

function parseJUnit(xml) {
  const suites = [...xml.matchAll(/<testsuite\b([^>]*)>/g)].map((m) => parseAttrs(m[1]));
  if (!suites.length) fail('pytest JUnit XML: no <testsuite> element');
  const total = (key) => suites.reduce((acc, s) => acc + Number(s[key] || 0), 0);
  const cases = [];
  for (const m of xml.matchAll(/<testcase\b([^>]*?)(?:\/>|>([\s\S]*?)<\/testcase>)/g)) {
    const attrs = parseAttrs(m[1]);
    const body = m[2] || '';
    let status = 'passed';
    if (/<failure\b/.test(body)) status = 'failed';
    else if (/<error\b/.test(body)) status = 'error';
    else if (/<skipped\b/.test(body)) status = 'skipped';
    cases.push({ classname: attrs.classname || '', name: attrs.name || '', status });
  }
  return {
    tests: total('tests'),
    failures: total('failures'),
    errors: total('errors'),
    skipped: total('skipped'),
    seconds: total('time'),
    cases,
  };
}

/** Minimal CSV reader (quoted fields supported): returns the named columns as row objects. */
function parseCsv(text, what, columns) {
  const rows = [];
  let field = '';
  let row = [];
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i += 1; } else if (ch === '"') quoted = false; else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  if (rows.length < 2) fail(`${what}: no data rows`);
  const header = rows[0].map((h) => h.replace(/^﻿/, '').trim());
  const index = columns.map((c) => {
    const k = header.indexOf(c);
    if (k < 0) fail(`${what}: missing column "${c}"`);
    return k;
  });
  return rows.slice(1).map((r) => Object.fromEntries(columns.map((c, j) => [c, r[index[j]]])));
}

function loadInputs(cfg) {
  const d = { cfg };
  const resultsDir = path.join(cfg.reportsDir, 'results');
  d.mc = readJson(path.join(resultsDir, 'model_comparison.json'), 'model comparison');
  d.results = {};
  d.refs = {};
  for (const t of TABLES) {
    const entry = pick(d.mc, `tables.${t}`, 'model_comparison.json');
    const models = pick(entry, 'models', `model_comparison.json tables.${t}`);
    if (!Array.isArray(models) || !models.length) fail(`model_comparison.json: tables.${t}.models is empty`);
    d.results[t] = {};
    for (const m of models) {
      const id = pick(m, 'model_id', `model_comparison.json tables.${t}.models[]`);
      const file = m.results_file || `${t}__${id}.json`;
      d.results[t][id] = readJson(path.join(resultsDir, file), `${t}/${id} evaluation results`);
    }
    d.refs[t] = {};
    for (const r of pick(entry, 'references', `model_comparison.json tables.${t}`)) {
      const kind = pick(r, 'kind', `model_comparison.json tables.${t}.references[]`);
      const file = r.results_file || `${t}__reference__${r.model_id}.json`;
      d.refs[t][kind] = { entry: r, result: readJson(path.join(resultsDir, file), `${t} ${kind} reference results`) };
    }
    for (const kind of ['ceiling', 'floor']) {
      if (!d.refs[t][kind]) fail(`model_comparison.json: tables.${t}.references has no "${kind}" entry`);
    }
  }
  d.junit = parseJUnit(readText(path.join(resultsDir, 'pytest_junit.xml'), 'pytest JUnit XML'));
  // optional: the slow suite (pytest -m slow), written by scripts/05_api_demo.py --run-slow-tests
  const slowFile = path.join(resultsDir, 'pytest_junit_slow.xml');
  d.junitSlow = fs.existsSync(slowFile) ? parseJUnit(readText(slowFile, 'slow-suite pytest JUnit XML')) : null;
  // optional: the A/B test of the ward_daily gc_tuned spec with its re-scores on further evaluation
  // seeds; without it (or if it cannot be read) the notes fall back to the single-seed caveat
  const abFile = path.join(cfg.reportsDir, 'experiments', 'ward_daily_band_groups_ab.json');
  d.abTest = null;
  if (fs.existsSync(abFile)) {
    try {
      d.abTest = readJson(abFile, 'ward_daily A/B test (reports/experiments)');
    } catch (err) {
      if (!(err instanceof InputError)) throw err;
      console.warn(`note: ${err.message}; the A/B evidence is left out of the notes`);
    }
  }

  const demoDir = path.join(cfg.reportsDir, 'api_demo');
  d.demo = readJson(path.join(demoDir, 'summary.json'), 'API demo summary');
  const steps = pick(d.demo, 'steps', 'api_demo/summary.json');
  const findStep = (label, predicate) => {
    const step = steps.find(predicate);
    if (!step) fail(`api_demo/summary.json: no step for ${label}`);
    const evidence = pick(step, 'evidence', `api_demo/summary.json step ${step.index}`);
    return { step, evidence: readJson(path.join(demoDir, evidence), `API demo evidence ${evidence}`) };
  };
  d.demoService = findStep('the service information (GET /)', (s) => s.method === 'GET' && s.path === '/');
  d.demoCatalogue = findStep('the scenario catalogue (GET /api/v1/scenarios)', (s) => s.method === 'GET'
    && s.path === '/api/v1/scenarios' && /catalogue/i.test(s.name || ''));
  d.demoGenerate = findStep('the high_occupancy generation (POST /api/v1/generate)', (s) => s.method === 'POST'
    && s.path === '/api/v1/generate' && /high_occupancy/.test(s.name || '') && s.status_code === 200);
  d.demoCalendar = findStep('the calendar-mode generation', (s) => s.method === 'POST'
    && s.path === '/api/v1/generate' && /calendar/i.test(s.name || '') && s.status_code === 200);

  d.registry = readJson(path.join(cfg.modelsDir, 'registry.json'), 'model registry');
  d.cards = {};
  for (const t of TABLES) {
    const models = pick(d.registry, `${t}.models`, 'models/registry.json');
    d.cards[t] = {};
    for (const id of Object.keys(models)) {
      d.cards[t][id] = readJson(path.join(cfg.modelsDir, t, id, 'model_card.json'), `${t}/${id} model card`);
    }
  }

  d.dq = readJson(path.join(cfg.processedDir, 'data_quality_report.json'), 'data quality report');
  d.catalog = readJson(path.join(cfg.processedDir, 'schema_catalog.json'), 'schema catalogue');
  // the clean reference ward-days (seasonal occupancy chart; same data as week1_occupancy_by_month.png)
  d.wardDailyClean = parseCsv(
    readText(path.join(cfg.processedDir, 'ward_daily_clean.csv'), 'clean ward_daily table'),
    'ward_daily_clean.csv', ['ward_id', 'month', 'occupancy_rate'],
  );

  if (fs.existsSync(cfg.uiScreenshot)) {
    d.uiScreenshot = { path: cfg.uiScreenshot, size: pngSize(cfg.uiScreenshot) };
  } else if (cfg.uiScreenshotRequired) {
    mustExist(cfg.uiScreenshot, 'UI screenshot');
  } else {
    console.warn('note: no UI screenshot found (assets/ui_screenshot.png); the closing slide is built without it');
  }
  return d;
}

// ------------------------------------------------------------------------------------------
// Formatting helpers
// ------------------------------------------------------------------------------------------
const NF = new Intl.NumberFormat('en-GB');
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const fmtInt = (n) => NF.format(Math.round(n));
const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
function pct(x, digits) {
  const v = Number(x) * 100;
  let d = digits;
  if (d === undefined) d = v !== 0 && Math.abs(v) < 1 ? 1 : 0;
  return `${v.toFixed(d)}%`;
}
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const word = (n) => (Number.isInteger(n) && n >= 0 && n < WORDS.length ? WORDS[n] : String(n));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const human = (id) => String(id).replace(/_/g, ' ');
const NBSP = ' ';
function listJoin(items, conj = 'and') {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} ${conj} ${items[items.length - 1]}`;
}
function segText(card) {
  if (!card.segment_columns || !card.segment_columns.length) return 'global';
  const label = card.segment_columns.map((c) => ({ ward_id: 'ward', admission_source: 'source' }[c] || human(c))).join(' × ');
  return `${card.n_segments} (${label})`;
}
/**
 * Readable name of an operational check: weekend/weekday ratio, "a–b relationship" for correlations
 * (the last name part is the second term: relationship_bed_pressure_cancellations ->
 * "bed pressure–cancellations relationship").
 */
function checkLabel(name) {
  const rel = /^relationship_(.+)$/.exec(name);
  if (rel) {
    const parts = rel[1].split('_');
    return parts.length >= 2 ? `${parts.slice(0, -1).join(' ')}–${parts[parts.length - 1]} relationship` : `${parts[0]} relationship`;
  }
  return human(name).replace(/\bweekend weekday\b/, 'weekend/weekday').replace(/\blos\b/, 'length of stay');
}
/** "1 failure", "2 failures". */
const plural = (n, noun) => `${fmtInt(n)} ${noun}${n === 1 ? '' : 's'}`;
/** "a, b and 2 more" for long lists. */
function shortList(items, max = 3) {
  return items.length > max ? `${items.slice(0, max).join(', ')} and ${word(items.length - max)} more` : listJoin(items);
}
/** A step of 1, 2 or 5 x 10^k that splits [0, max] into at most `parts` intervals. */
function niceStep(max, parts = 3) {
  const raw = Math.max(max, 1e-9) / parts;
  const p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((k) => k * p).find((s) => s >= raw);
}
function condText(conditions) {
  return Object.entries(conditions || {}).map(([col, vals]) => `${col} ∈ {${vals.join(', ')}}`).join('; ');
}
function filterText(filters) {
  const parts = [];
  for (const [col, spec] of Object.entries(filters || {})) {
    if (spec.min !== undefined) parts.push(`${col} ≥ ${spec.min}`);
    if (spec.max !== undefined) parts.push(`${col} ≤ ${spec.max}`);
    if (spec.in !== undefined) parts.push(`${col} ∈ {${spec.in.join(', ')}}`);
    if (spec.not_in !== undefined) parts.push(`${col} ∉ {${spec.not_in.join(', ')}}`);
  }
  return parts.join('; ');
}

// ------------------------------------------------------------------------------------------
// Derived context: every number the slides show
// ------------------------------------------------------------------------------------------
/**
 * Seed evidence of the ward_daily A/B test (reports/experiments/ward_daily_band_groups_ab.json):
 * the previous and the adopted spec scored on the same evaluation seeds. It is used only when it
 * describes the current default model - the test's control is the default, its official run used
 * this evaluation's seed and the adopted spec's composite on that seed equals the default's
 * composite in model_comparison.json - otherwise the notes keep the single-seed caveat (null).
 * Ranges and means are computed from the per-seed lists, not read from the stored summaries.
 */
function abEvidence(ab, ctx) {
  const t = 'ward_daily';
  const skip = (why) => {
    console.warn(`note: the ward_daily A/B test (reports/experiments) is left out of the notes: ${why}`);
    return null;
  };
  if ((ab.table || t) !== t) return skip(`it is about table "${ab.table}"`);
  const prev = opt(ab, 'summary.previous_spec');
  const adopt = opt(ab, 'summary.adopted_spec');
  if (!prev || !adopt) return skip('no summary.previous_spec / summary.adopted_spec');
  const seeds = adopt.evaluation_seeds;
  const n = Array.isArray(seeds) ? seeds.length : 0;
  const numbers = (xs) => Array.isArray(xs) && xs.length === n && xs.every((x) => typeof x === 'number' && Number.isFinite(x));
  if (n < 2 || !numbers(seeds)) return skip('summary.adopted_spec.evaluation_seeds lists fewer than two seeds');
  if (!numbers(prev.evaluation_seeds) || prev.evaluation_seeds.some((s, i) => s !== seeds[i])) return skip('the two specs were not scored on the same seeds');
  for (const [name, spec] of [['previous_spec', prev], ['adopted_spec', adopt]]) {
    for (const key of ['composite', 'raw_hard_valid', 'operational_checks_passed']) {
      if (!numbers(spec[key])) return skip(`summary.${name}.${key} is not a list of ${n} numbers`);
    }
    if (typeof spec.operational_checks_scored !== 'number') return skip(`summary.${name}.operational_checks_scored is missing`);
  }
  if (prev.operational_checks_scored !== adopt.operational_checks_scored) return skip('the two specs were scored on different numbers of checks');
  const seed = opt(ab, 'official_run.evaluation_seed');
  const k = seeds.indexOf(seed);
  if (seed !== ctx.runConfig[t].seed || k < 0) return skip(`its official evaluation seed (${seed}) is not this evaluation's seed (${ctx.runConfig[t].seed})`);
  const variants = opt(ab, 'official_run.variants', []);
  const role = (r) => (Array.isArray(variants) ? variants.find((v) => v && v.role === r) || {} : {}).variant;
  if (role('control') !== ctx.defaults[t]) return skip(`its control (${role('control')}) is not the default model ${ctx.defaults[t]}`);
  const current = (ctx.cmp[t].models.find((m) => m.id === ctx.defaults[t]) || {}).composite;
  if (typeof current !== 'number' || Math.abs(current - adopt.composite[k]) > 5e-4) {
    return skip(`the adopted spec scored ${adopt.composite[k]} on seed ${seed}, the default model ${current}`);
  }
  const stats = (xs) => ({ min: Math.min(...xs), max: Math.max(...xs), mean: sum(xs) / xs.length });
  const scored = adopt.operational_checks_scored;
  const extra = seeds.map((s, i) => i).filter((i) => i !== k);
  const failing = extra.filter((i) => adopt.operational_checks_passed[i] < scored);
  // names of the checks the adopted spec fails on the extra seeds (robustness screens, if listed)
  const bySeed = opt(ab, 'robustness_screens.by_seed', {});
  const failedChecks = [...new Set(failing.flatMap((i) => {
    const rows = Array.isArray(bySeed[String(seeds[i])]) ? bySeed[String(seeds[i])] : [];
    return (rows.find((r) => r && r.variant === role('adopted')) || {}).failed_checks || [];
  }))];
  return {
    n,
    seed,
    prev: { composite: stats(prev.composite), valid: stats(prev.raw_hard_valid), passed: stats(prev.operational_checks_passed) },
    adopt: { composite: stats(adopt.composite), valid: stats(adopt.raw_hard_valid), passed: stats(adopt.operational_checks_passed) },
    scored,
    extraSeeds: extra.length,
    failingSeeds: failing.length,
    maxFailing: failing.length ? Math.max(...failing.map((i) => scored - adopt.operational_checks_passed[i])) : 0,
    failedChecks,
  };
}

function buildContext(d) {
  const ctx = {};
  const cat = (t) => pick(d.catalog, `tables.${t}`, 'schema_catalog.json');

  // Hospital and data
  ctx.wards = pick(cat('ward_daily'), 'wards', 'schema_catalog.json ward_daily').map((w) => ({
    id: pick(w, 'ward_id', 'schema_catalog.json ward'),
    name: w.name || w.ward_id,
    beds: num(w.bed_capacity, `schema_catalog.json ward ${w.ward_id} bed_capacity`),
  }));
  ctx.nBeds = sum(ctx.wards.map((w) => w.beds));
  ctx.nWards = ctx.wards.length;
  const range = pick(d.dq, 'consistency_checks.ward_daily_continuity.date_range', 'data_quality_report.json');
  // read the calendar year from the ISO text itself: Date() would apply the local time zone
  const yearOf = (s) => (/^(\d{4})-\d{2}-\d{2}/.exec(String(s)) || [])[1];
  ctx.year0 = Number(yearOf(range[0]));
  ctx.year1 = Number(yearOf(range[1]));
  if (!ctx.year0 || !ctx.year1) fail(`data_quality_report.json: unreadable date_range ${JSON.stringify(range)}`);
  ctx.nYears = ctx.year1 - ctx.year0 + 1;

  // Seasonal profile of the reference data: mean midnight occupancy per ward and calendar month
  {
    const acc = {};
    const all = {};
    for (const r of d.wardDailyClean) {
      const m = Number(r.month);
      const v = Number(r.occupancy_rate);
      if (!Number.isInteger(m) || m < 1 || m > 12 || !Number.isFinite(v) || r.occupancy_rate === '') continue;
      const key = `${r.ward_id}|${m}`;
      acc[key] = acc[key] || [0, 0];
      acc[key][0] += v;
      acc[key][1] += 1;
      all[m] = all[m] || [0, 0];
      all[m][0] += v;
      all[m][1] += 1;
    }
    const months = Object.keys(all).map(Number).sort((a, b) => a - b);
    if (months.length < 2) fail('ward_daily_clean.csv: no usable month / occupancy_rate values');
    const byWard = {};
    for (const w of ctx.wards) {
      const vals = months.map((m) => (acc[`${w.id}|${m}`] ? acc[`${w.id}|${m}`][0] / acc[`${w.id}|${m}`][1] : null));
      if (vals.some((v) => v !== null)) byWard[w.id] = vals;
    }
    const finite = (xs) => xs.filter((v) => typeof v === 'number');
    const swing = Object.entries(byWard).map(([id, vals]) => ({ id, swing: Math.max(...finite(vals)) - Math.min(...finite(vals)) }))
      .sort((a, b) => b.swing - a.swing);
    const overall = months.map((m) => all[m][0] / all[m][1]);
    const peak = months[overall.indexOf(Math.max(...overall))];
    const low = months[overall.indexOf(Math.min(...overall))];
    const allVals = Object.values(byWard).flatMap(finite);
    ctx.season = {
      months,
      byWard,
      strongest: swing.slice(0, 2).map((s) => s.id),
      peak,
      low,
      winterPeakSummerLow: [12, 1, 2].includes(peak) && [6, 7, 8].includes(low),
      min: Math.min(...allVals),
      max: Math.max(...allVals),
    };
  }
  ctx.dq = {};
  ctx.rules = {};
  ctx.cols = {};
  for (const t of TABLES) {
    const s = pick(d.dq, `summary.${t}`, 'data_quality_report.json');
    const tab = pick(d.dq, `tables.${t}`, 'data_quality_report.json');
    const injectedByType = pick(tab, 'issues_injected', `data_quality_report.json tables.${t}`);
    const detectedByType = pick(tab, 'issues_detected', `data_quality_report.json tables.${t}`);
    const injected = sum(Object.values(injectedByType).map((o) => sum(Object.values(o))));
    const detected = num(pick(s, 'issues_detected', `data_quality_report.json summary.${t}`), `issues_detected ${t}`);
    // per issue type, detected duplicates merged like the injection log (as in reporting/week1.py)
    const typeTotals = (byType, merge) => {
      const out = {};
      for (const [type, cols] of Object.entries(byType || {})) {
        const key = merge && DUPLICATE_KINDS.includes(type) ? 'duplicate_row' : type;
        out[key] = (out[key] || 0) + sum(Object.values(cols || {}).map(Number));
      }
      return out;
    };
    const injectedTypes = typeTotals(injectedByType, false);
    const detectedTypes = typeTotals(detectedByType, true);
    const everyTypeFound = Object.entries(injectedTypes).every(([type, n]) => (detectedTypes[type] || 0) >= n);
    ctx.dq[t] = {
      rawRows: num(s.raw_rows, `raw_rows ${t}`),
      finalRows: num(s.final_rows, `final_rows ${t}`),
      retention: num(s.retention_rate, `retention_rate ${t}`),
      detected,
      injected,
      injectedTypes,
      detectedTypes,
      everyTypeFound,
      resolved: num(s.injected_issues_resolved_rate, `injected_issues_resolved_rate ${t}`),
      exact: num(s.injected_issues_exact_recovery_rate, `injected_issues_exact_recovery_rate ${t}`),
      validRaw: num(s.hard_valid_rate_raw, `hard_valid_rate_raw ${t}`),
      validFinal: num(s.hard_valid_rate_final, `hard_valid_rate_final ${t}`),
      cellAgreement: num(s.cell_agreement_with_truth, `cell_agreement_with_truth ${t}`),
      train: num(s.train_rows, `train_rows ${t}`),
      holdout: num(s.holdout_rows, `holdout_rows ${t}`),
      dropped: num(s.rows_dropped, `rows_dropped ${t}`),
      holdoutFraction: num(pick(tab, 'split.holdout_fraction', `data_quality_report.json tables.${t}`), 'holdout_fraction'),
      stratifyBy: pick(tab, 'split.stratify_by', `data_quality_report.json tables.${t}`),
    };
    const rules = pick(cat(t), 'rules', `schema_catalog.json ${t}`);
    ctx.rules[t] = {
      hard: rules.filter((r) => r.severity === 'hard').length,
      soft: rules.filter((r) => r.severity === 'soft').length,
    };
    ctx.cols[t] = {
      total: pick(cat(t), 'columns', `schema_catalog.json ${t}`).length,
      training: pick(cat(t), 'training_columns', `schema_catalog.json ${t}`).length,
      derived: pick(cat(t), 'derived_columns', `schema_catalog.json ${t}`).length,
    };
  }
  ctx.hardRules = ctx.rules.ward_daily.hard + ctx.rules.patient_stays.hard;
  ctx.weekdays = pick(cat('ward_daily').columns.find((c) => c.name === 'day_of_week') || {}, 'categories', 'schema_catalog.json day_of_week');
  const bandDesc = pick(cat('ward_daily').columns.find((c) => c.name === 'occupancy_level') || {}, 'description', 'schema_catalog.json occupancy_level');
  ctx.bandThresholds = [...bandDesc.matchAll(/(low|normal|high|critical)?\s*<\s*([0-9.]+)/g)].map((m) => Number(m[2]));
  ctx.bandNames = pick(cat('ward_daily').columns.find((c) => c.name === 'occupancy_level') || {}, 'categories', 'schema_catalog.json occupancy_level');
  if (ctx.bandThresholds.length !== ctx.bandNames.length - 1) fail(`schema_catalog.json: cannot read the occupancy band thresholds from "${bandDesc}"`);

  // Constraint groups (built-in vs programmable) from the catalogue
  ctx.groupKind = {};
  for (const t of TABLES) {
    for (const c of pick(cat(t), 'constraints', `schema_catalog.json ${t}`)) ctx.groupKind[c.group] = c.programmable ? 'programmable' : 'built-in';
  }
  ctx.groupRules = {};
  for (const t of TABLES) {
    for (const c of cat(t).constraints) {
      ctx.groupRules[c.group] = [...new Set([...(ctx.groupRules[c.group] || []), ...(c.enforces_rules || [])])];
    }
  }

  // Models: registry + model cards
  ctx.defaults = {};
  ctx.experiments = [];
  for (const t of TABLES) {
    ctx.defaults[t] = pick(d.registry, `${t}.default`, 'models/registry.json');
    for (const id of Object.keys(d.cards[t])) if (!ctx.experiments.includes(id)) ctx.experiments.push(id);
  }
  ctx.experiments.sort((a, b) => {
    const ia = PREFERRED_ORDER.indexOf(a);
    const ib = PREFERRED_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  for (const id of ABLATION) {
    for (const t of TABLES) if (!d.cards[t][id]) fail(`model card ${t}/${id} is needed for the ablation slide`);
  }
  ctx.allGroups = [];
  for (const t of TABLES) {
    for (const id of Object.keys(d.cards[t])) {
      for (const g of d.cards[t][id].constraint_groups || []) if (!ctx.allGroups.includes(g)) ctx.allGroups.push(g);
    }
  }
  for (const g of ctx.allGroups) if (!ctx.groupKind[g]) ctx.groupKind[g] = 'programmable';
  const epochs = [];
  for (const t of TABLES) {
    for (const id of Object.keys(d.cards[t])) {
      const e = opt(d.cards[t][id], 'training.sdv_params.epochs');
      if (e !== null && !epochs.includes(e)) epochs.push(e);
    }
  }
  ctx.epochs = epochs;

  // Evaluation
  ctx.weights = pick(d.mc, 'composite_weights', 'model_comparison.json');
  ctx.runConfig = {};
  ctx.cmp = {};
  for (const t of TABLES) {
    const entry = d.mc.tables[t];
    ctx.runConfig[t] = opt(entry, 'run_config', opt(d.mc, 'run_config', {}));
    const models = entry.models
      .filter((m) => m.status === 'ok' && typeof m.composite === 'number')
      .map((m) => ({
        id: m.model_id,
        composite: m.composite,
        components: pick(m, 'components', `model_comparison.json ${t}/${m.model_id}`),
        normalised: opt(m, 'normalised.composite'),
      }))
      .sort((a, b) => b.composite - a.composite);
    if (!models.length) fail(`model_comparison.json: no scored model for ${t}`);
    const ceiling = d.refs[t].ceiling.entry;
    const floor = d.refs[t].floor.entry;
    ctx.cmp[t] = {
      models,
      skipped: entry.models.filter((m) => !(m.status === 'ok' && typeof m.composite === 'number')).map((m) => m.model_id),
      best: pick(entry, 'best_model', `model_comparison.json tables.${t}`),
      ceiling: num(ceiling.composite, `${t} ceiling composite`),
      floor: num(floor.composite, `${t} floor composite`),
      ceilingComponents: ceiling.components,
      floorComponents: floor.components,
      noise: opt(entry, 'noise_calibration', {}),
    };
  }
  ctx.quick = Boolean(opt(d.mc, 'run_config.quick', false)) || TABLES.some((t) => Boolean(ctx.runConfig[t].quick));
  const evalRows = [...new Set(TABLES.map((t) => ctx.runConfig[t].rows).filter((v) => typeof v === 'number'))];
  const epochText = ctx.epochs.length ? `${ctx.epochs.join('/')}-epoch CTGAN/TVAE` : 'quick-trained models';
  const rowsText = evalRows.length ? `${evalRows.map(fmtInt).join('/')} evaluated rows` : 'reduced evaluation';
  ctx.quickNote = ctx.quick ? `Quick-run results (${epochText}, ${rowsText})` : '';
  ctx.evalRowsText = evalRows.map(fmtInt).join('/');
  // evaluation seeds of the stored run configurations (one seed: the limitations notes say so)
  ctx.evalSeeds = [...new Set(TABLES.map((t) => ctx.runConfig[t].seed).filter((v) => typeof v === 'number'))];

  // Scenarios (catalogue from the API demo evidence + scenario coverage of the default models)
  const catBody = pick(d.demoCatalogue.evidence, 'response.body', 'scenario catalogue evidence');
  ctx.categories = pick(catBody, 'categories', 'scenario catalogue evidence');
  ctx.scenarios = pick(catBody, 'scenarios', 'scenario catalogue evidence');
  ctx.coverage = {};
  for (const t of TABLES) {
    const res = d.results[t][ctx.defaults[t]];
    if (!res) fail(`no evaluation results for the default model ${t}/${ctx.defaults[t]}`);
    ctx.coverage[t] = pick(res, 'scenario_coverage', `${t}/${ctx.defaults[t]} results`);
  }
  ctx.nScenarios = ctx.scenarios.length;
  ctx.familiesCovered = Object.keys(ctx.categories).filter((c) => c !== 'baseline' && ctx.scenarios.some((s) => s.category === c));
  ctx.nFamilies = Object.keys(ctx.categories).filter((c) => c !== 'baseline').length;

  // facts several slides state, checked on the data instead of assumed
  ctx.allOutputValid = TABLES.every((t) => d.mc.tables[t].models.filter((m) => m.status === 'ok').every((m) => m.output_validity === 1));
  ctx.aggMirror = TABLES.every((t) => opt(d.results[t][ctx.defaults[t]], 'privacy.aggregate_disclosure.mirrors_training_frequencies') === true);
  ctx.sdvVersion = opt(d.cards.ward_daily[ctx.defaults.ward_daily] || {}, 'sdv_version', opt(d.demo, 'sdv_version', null));

  // utility families of the default models below the no-learning floor (slides 16 and 19 use this list)
  const famScore = (t, kind, key) => (kind === 'model' ? opt(d.results[t][ctx.defaults[t]], `${key}.score`) : opt(d.refs[t][kind].result, `${key}.score`));
  ctx.famScore = famScore;
  ctx.utilityFamilies = [['ML utility · ward_daily', 'ward_daily', 'ml_utility'], ['ML utility · patient_stays', 'patient_stays', 'ml_utility'], ['Optimisation · ward_daily', 'ward_daily', 'optimization']]
    .filter(([, t, k]) => typeof famScore(t, 'model', k) === 'number');
  ctx.utilityGaps = ctx.utilityFamilies.flatMap(([label, t, k]) => {
    const m = famScore(t, 'model', k);
    const f = famScore(t, 'floor', k);
    return typeof f === 'number' && m < f ? [{ label, t, k, m, f, level: f3(m) === f3(f) }] : [];
  });
  // composite families of the default models that score below the no-learning floor (privacy and
  // validity are left out: the floor copies nothing and is not a validity target)
  ctx.componentGaps = TABLES.flatMap((t) => {
    const m = ctx.cmp[t].models.find((x) => x.id === ctx.defaults[t]);
    const fl = ctx.cmp[t].floorComponents || {};
    if (!m) return [];
    return ['fidelity', 'operational', 'utility']
      // a utility family already listed below the floor (ctx.utilityGaps) is not repeated here
      .filter((f) => !(f === 'utility' && ctx.utilityGaps.some((g) => g.t === t)))
      .filter((f) => typeof m.components[f] === 'number' && typeof fl[f] === 'number' && m.components[f] < fl[f])
      .map((f) => ({ t, f, m: m.components[f], floor: fl[f] }));
  });
  // operational checks of the default models: failing, and passing but beyond real-data noise
  ctx.opGaps = Object.fromEntries(TABLES.map((t) => {
    const op = opt(d.results[t][ctx.defaults[t]], 'operational', {});
    const failedChecks = op.failed_checks || [];
    return [t, { failed: failedChecks, beyondNoise: (op.beyond_real_noise || []).filter((c) => !failedChecks.includes(c)) }];
  }));
  // weakest scenario scores of the default models (both tables)
  const scored = TABLES.flatMap((t) => (ctx.coverage[t].scenarios || []).filter((s) => typeof s.score === 'number'));
  ctx.weakestScenarios = scored.sort((a, b) => a.score - b.score).slice(0, 2);
  ctx.weakFiltered = ctx.weakestScenarios.length > 0
    && ctx.weakestScenarios.every((s) => Object.keys((ctx.scenarios.find((c) => c.name === s.name) || {}).filters || {}).length > 0);
  // reference support of the scenarios in the training split (catalogue served with include_support)
  const supported = ctx.scenarios.filter((s) => s.category !== 'baseline' && typeof opt(s, 'support.matching_rows') === 'number');
  ctx.support = supported.length ? {
    min: supported.reduce((a, b) => (b.support.matching_rows < a.support.matching_rows ? b : a)),
    max: supported.reduce((a, b) => (b.support.matching_rows > a.support.matching_rows ? b : a)),
    split: opt(supported[0], 'support.split', 'train'),
  } : null;
  // ward_daily marginals of the default model grouped by a recomputed band (model card)
  ctx.bandGroups = [...new Set((opt(d.cards.ward_daily[ctx.defaults.ward_daily] || {}, 'constraints', []))
    .flatMap((c) => c.band_groups || []))];
  // re-scores of the ward_daily default on further evaluation seeds (optional A/B test file)
  ctx.ab = d.abTest ? abEvidence(d.abTest, ctx) : null;

  // API demo
  ctx.endpoints = pick(d.demoService.evidence, 'response.body.endpoints', 'service info evidence');
  ctx.demoSteps = pick(d.demo, 'steps', 'api_demo/summary.json');
  return ctx;
}

// ------------------------------------------------------------------------------------------
// Design system
// ------------------------------------------------------------------------------------------
const THEME = {
  name: 'BedSynth Ward',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '17262B', lt1: 'FFFFFF', dk2: '0E3B43', lt2: 'EAF3F2',
    accent1: '127C76', accent2: 'D17A22', accent3: '5E9C96', accent4: '7D8E93', accent5: 'B5452F', accent6: '2F6690',
    hlink: '127C76', folHlink: '0E3B43',
  },
};
const HEX = Object.assign({}, THEME.colors, { muted: '55656A', rule: 'C9D8D6', tintAmber: 'FBEEE1', tintRed: 'F7E6E2' });
const C = {
  text: 'tx1', ink: 'tx2', bg: 'bg1', tint: 'bg2',
  teal: 'accent1', amber: 'accent2', tealMid: 'accent3', grey: 'accent4', red: 'accent5', steel: 'accent6',
};
const CHART_FONT = '+mn-lt'; // theme body font (chart text is otherwise hard-coded Arial)
const HEAD_FONT = '+mj-lt'; // theme heading font, for display text outside the title placeholders
const CODE_FONT = 'Consolas';
// eight distinguishable series colours (Okabe-Ito based), each at least 3:1 against the white slide
const WARD_COLORS = ['0072B2', 'C47F00', '009E73', 'CC79A7', '4A9AD0', 'D55E00', '6A3D9A', '17262B'];
const W = 13.333;
const H = 7.5;
const MX = 0.6;
const CW = W - 2 * MX;
const TOP = 1.72;

function createPresentation() {
  const pres = new PptxGenJS();
  pres.layout = 'LAYOUT_WIDE';
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = 'BedSynth AI - Synthetic Hospital Operations Data Generator';
  pres.subject = 'Final presentation of the BedSynth AI university team project';
  pres.author = 'BedSynth AI team';
  pres.company = 'BedSynth AI team';

  pres.defineSlideMaster({
    title: 'BS_TITLE_DARK',
    background: { color: C.ink },
    objects: [
      {
        placeholder: {
          options: { name: 'title', type: 'title', x: 0.8, y: 1.05, w: 11.7, h: 1.05, fontSize: 54, bold: true, color: C.bg, align: 'left', valign: 'top', margin: 0 },
          text: '',
        },
      },
    ],
  });
  pres.defineSlideMaster({
    title: 'BS_CONTENT',
    background: { color: C.bg },
    objects: [
      {
        placeholder: {
          options: { name: 'title', type: 'title', x: MX, y: 0.74, w: 10.2, h: 0.72, fontSize: 30, bold: true, color: C.text, align: 'left', valign: 'top', margin: 0 },
          text: '',
        },
      },
      { text: { text: 'BedSynth AI · synthetic hospital operations data', options: { x: MX, y: 7.02, w: 6, h: 0.26, fontSize: 10, color: HEX.muted, margin: 0, valign: 'middle' } } },
    ],
    slideNumber: { x: W - MX - 0.8, y: 7.02, w: 0.8, h: 0.26, fontSize: 10, color: HEX.muted, align: 'right', margin: 0, valign: 'middle' },
  });
  return pres;
}

// ------------------------------------------------------------------------------------------
// Drawing helpers
// ------------------------------------------------------------------------------------------
function makeKit(pres) {
  const S = pres.ShapeType;
  const kit = {
    S,
    text(slide, value, o) {
      slide.addText(value, Object.assign({ isTextBox: true, margin: 0, valign: 'top', color: C.text, fontSize: 14, fit: 'none' }, o));
    },
    box(slide, o) {
      const radius = o.radius === undefined ? 0.1 : o.radius;
      const shape = radius ? S.roundRect : S.rect;
      const props = {
        x: o.x, y: o.y, w: o.w, h: o.h,
        fill: { color: o.fill || C.tint, transparency: o.transparency || 0 },
        line: o.line ? { color: o.line, width: o.lineWidth || 1 } : { type: 'none' },
        objectName: o.name,
      };
      if (radius) props.rectRadius = radius;
      if (o.shadow) props.shadow = { type: 'outer', color: '000000', opacity: 0.12, blur: 6, offset: 2, angle: 90 };
      slide.addShape(shape, props);
    },
    line(slide, x1, y1, x2, y2, o = {}) {
      slide.addShape(S.line, {
        x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
        flipH: x2 < x1, flipV: y2 < y1,
        line: { color: o.color || HEX.muted, width: o.width || 1.5, endArrowType: o.arrow ? 'triangle' : undefined, dashType: o.dash || 'solid' },
      });
    },
    circle(slide, x, y, dia, fill, o = {}) {
      slide.addShape(S.ellipse, { x, y, w: dia, h: dia, fill: { color: fill }, line: o.line ? { color: o.line, width: 1.25 } : { type: 'none' }, objectName: o.name });
    },
    /** Icon in a coloured circle (preset shapes only, so it renders everywhere). */
    icon(slide, kind, x, y, dia, fill = C.teal, glyph = C.bg) {
      kit.circle(slide, x, y, dia, fill, { name: `Icon ${kind}` });
      const g = (fx, fy, fw, fh) => ({ x: x + dia * fx, y: y + dia * fy, w: dia * fw, h: dia * fh });
      const solid = { fill: { color: glyph }, line: { type: 'none' } };
      switch (kind) {
        case 'hospital':
          slide.addShape(S.mathPlus, Object.assign(g(0.18, 0.18, 0.64, 0.64), solid));
          break;
        case 'data':
          slide.addShape(S.can, Object.assign(g(0.3, 0.24, 0.4, 0.52), solid));
          break;
        case 'model':
          slide.addShape(S.cube, Object.assign(g(0.27, 0.25, 0.46, 0.48), solid));
          break;
        case 'engine':
          slide.addShape(S.gear6, Object.assign(g(0.22, 0.22, 0.56, 0.56), solid));
          break;
        case 'alert':
          slide.addShape(S.lightningBolt, Object.assign(g(0.27, 0.2, 0.46, 0.6), solid));
          break;
        case 'lock':
          slide.addShape(S.blockArc, Object.assign(g(0.33, 0.2, 0.34, 0.42), { fill: { color: glyph }, line: { type: 'none' }, angleRange: [180, 0], arcThicknessRatio: 0.32 }));
          slide.addShape(S.roundRect, Object.assign(g(0.27, 0.42, 0.46, 0.36), solid, { rectRadius: dia * 0.05 }));
          break;
        case 'check':
          slide.addShape(S.custGeom, Object.assign(g(0.2, 0.22, 0.6, 0.56), solid, {
            points: [[0.0, 0.52], [0.16, 0.36], [0.38, 0.58], [0.84, 0.1], [1.0, 0.26], [0.38, 0.9]]
              .map(([px, py]) => ({ x: dia * 0.6 * px, y: dia * 0.56 * py }))
              .concat([{ close: true }]),
          }));
          break;
        case 'chart':
          [[0.24, 0.5, 0.26], [0.42, 0.32, 0.44], [0.6, 0.42, 0.34]].forEach(([fx, fy, fh]) => {
            slide.addShape(S.rect, Object.assign(g(fx, fy, 0.14, fh), solid));
          });
          break;
        case 'tiles':
          [[0.25, 0.25], [0.53, 0.25], [0.25, 0.53], [0.53, 0.53]].forEach(([fx, fy]) => {
            slide.addShape(S.rect, Object.assign(g(fx, fy, 0.22, 0.22), solid));
          });
          break;
        case 'api':
          kit.text(slide, '</>', Object.assign(g(0, 0, 1, 1), { fontFace: CODE_FONT, bold: true, color: glyph, fontSize: Math.round(dia * 26), align: 'center', valign: 'middle' }));
          break;
        case 'future':
          slide.addShape(S.chevron, Object.assign(g(0.22, 0.26, 0.3, 0.48), solid));
          slide.addShape(S.chevron, Object.assign(g(0.46, 0.26, 0.3, 0.48), solid));
          break;
        default:
          kit.text(slide, String(kind).charAt(0).toUpperCase(), Object.assign(g(0, 0, 1, 1), { bold: true, color: glyph, fontSize: Math.round(dia * 28), align: 'center', valign: 'middle' }));
      }
    },
    /** Big number with a label underneath. */
    stat(slide, x, y, w, value, label, o = {}) {
      kit.text(slide, value, { x, y, w, h: o.valueH || 0.66, fontSize: o.size || 36, bold: true, color: o.color || C.teal, fontFace: HEAD_FONT, valign: 'bottom' });
      kit.text(slide, label, { x, y: y + (o.valueH || 0.66) + 0.04, w, h: o.labelH || 0.5, fontSize: o.labelSize || 14, color: o.labelColor || HEX.muted });
    },
  };
  return kit;
}

const chartBase = () => ({
  catAxisLabelFontFace: CHART_FONT, catAxisLabelFontSize: 12, catAxisLabelColor: HEX.dk1,
  valAxisLabelFontFace: CHART_FONT, valAxisLabelFontSize: 12, valAxisLabelColor: HEX.muted,
  dataLabelFontFace: CHART_FONT, dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
  legendFontFace: CHART_FONT, legendFontSize: 12, legendColor: HEX.dk1,
  titleFontFace: CHART_FONT, titleFontSize: 14, titleColor: HEX.dk1, titleBold: true,
  catAxisLineShow: true, catAxisLineColor: HEX.rule, valAxisLineShow: false,
  catGridLine: { style: 'none' },
  valGridLine: { color: 'E3ECEB', size: 0.75, style: 'solid' },
});

// ------------------------------------------------------------------------------------------
// Slides
// ------------------------------------------------------------------------------------------
function buildSlides(pres, d, ctx) {
  const K = makeKit(pres);
  // one PowerPoint section per speaking part; the split and the target times follow the presentation
  // plan of docs/TEAM_PLAN.md (section 5): A 2 + 3 min (slides 1-4, 5-7), B 4 min, C 4 min, D 2 min of
  // slides plus about 3 min of live demo, C 2 min to close - about 17 minutes of slides in total
  const sections = {
    A: named('Opening, data and scenarios - Member A'),
    B: named('SDV pipeline - Member B'),
    C: named('Evaluation - Member C'),
    D: named('API, demo and testing - Member D'),
    close: named('Limitations and close - Member C'),
  };
  const opened = new Set();
  const ensureSection = (key) => {
    if (!opened.has(key)) {
      pres.addSection({ title: sections[key] });
      opened.add(key);
    }
    return sections[key];
  };
  const notesFor = (member, seconds, body, extra = '') => named(`Suggested speaker: Member ${member}. Target time: about ${seconds} seconds${extra ? `, ${extra}` : ''}.\n\n${body}`);

  function contentSlide({ member, section, kicker, title, notes, seconds = 75, timeExtra = '', footnote = '', source = '' }) {
    const slide = pres.addSlide({ masterName: 'BS_CONTENT', sectionTitle: ensureSection(section || member) });
    K.text(slide, kicker.toUpperCase(), { x: MX, y: 0.42, w: 9, h: 0.28, fontSize: 12, bold: true, color: C.teal, charSpacing: 1.5, objectName: 'Kicker' });
    slide.addText(title, { placeholder: 'title' });
    // speaker pill (top right)
    const pw = 1.62;
    const px = W - MX - pw;
    K.box(slide, { x: px, y: 0.4, w: pw, h: 0.42, radius: 0.21, fill: C.tint, name: 'Speaker' });
    K.circle(slide, px + 0.05, 0.44, 0.34, C.teal);
    K.text(slide, member, { x: px + 0.05, y: 0.44, w: 0.34, h: 0.34, fontSize: 14, bold: true, color: C.bg, align: 'center', valign: 'middle' });
    K.text(slide, MEMBERS[member], { x: px + 0.47, y: 0.4, w: pw - 0.52, h: 0.42, fontSize: 14, color: C.text, valign: 'middle' });
    const foot = [source, footnote].filter(Boolean).join('   ·   ');
    if (foot) K.text(slide, foot, { x: MX, y: 6.74, w: CW, h: 0.24, fontSize: 10, italic: true, color: HEX.muted, valign: 'middle', objectName: 'Footnote' });
    slide.addNotes(notesFor(member, seconds, notes, timeExtra));
    return slide;
  }

  const quick = ctx.quickNote;
  const wd = ctx.dq.ward_daily;
  const ps = ctx.dq.patient_stays;
  const dWD = ctx.defaults.ward_daily;
  const dPS = ctx.defaults.patient_stays;
  const sameDefault = dWD === dPS;
  const resWD = d.results.ward_daily[dWD];
  const resPS = d.results.patient_stays[dPS];
  const opWD = pick(resWD, 'operational', `ward_daily/${dWD} results`);
  const opPS = pick(resPS, 'operational', `patient_stays/${dPS} results`);
  const nExperiments = ctx.experiments.length;
  const nGroups = ctx.allGroups.length;
  const progGroups = ctx.allGroups.filter((g) => ctx.groupKind[g] === 'programmable');
  const builtinGroups = ctx.allGroups.filter((g) => ctx.groupKind[g] !== 'programmable');
  const families = Object.keys(ctx.weights);
  const famLabel = { fidelity: 'Fidelity', validity: 'Raw validity', operational: 'Operational patterns', utility: 'Utility', privacy: 'Privacy' };

  // ---------------------------------------------------------------- 1. Title
  {
    const slide = pres.addSlide({ masterName: 'BS_TITLE_DARK', sectionTitle: ensureSection('A') });
    K.text(slide, 'UNIVERSITY TEAM PROJECT  ·  FINAL PRESENTATION', { x: 0.8, y: 0.62, w: 9, h: 0.32, fontSize: 14, bold: true, color: HEX.lt2, charSpacing: 1.5 });
    slide.addText('BedSynth AI', { placeholder: 'title' });
    K.text(slide, 'Synthetic hospital operations data generator', { x: 0.8, y: 2.08, w: 11.5, h: 0.55, fontSize: 28, color: C.bg, fontFace: HEAD_FONT });
    K.text(slide, 'SDV-generated bed-occupancy and scheduling data for testing optimisation models,\nwithout exposing real facility data', { x: 0.8, y: 2.74, w: 11.2, h: 0.82, fontSize: 18, color: HEX.lt2 });
    K.text(slide, ['A', 'B', 'C', 'D'].map((k) => MEMBERS[k]).join('  ·  '), { x: 0.8, y: 3.72, w: 11.7, h: 0.36, fontSize: 16, bold: true, color: C.bg });

    // Ward bed board: one square per funded bed, filled share = mean midnight occupancy (real data)
    const occ = opt(opWD.checks.find((c) => c.name === 'occupancy_mean_by_ward') || {}, 'real', {});
    const cols = 5;
    const tile = 0.165;
    const pitch = 0.21;
    const blockW = cols * pitch - (pitch - tile);
    const gap = (11.73 - ctx.nWards * blockW) / Math.max(1, ctx.nWards - 1);
    const baseY = 6.08;
    ctx.wards.forEach((w, i) => {
      const x0 = 0.8 + i * (blockW + gap);
      const share = typeof occ[w.id] === 'number' ? occ[w.id] : 0;
      const filled = Math.round(w.beds * share);
      for (let k = 0; k < w.beds; k += 1) {
        const row = Math.floor(k / cols);
        const col = k % cols;
        slide.addShape(pres.ShapeType.rect, {
          x: x0 + col * pitch, y: baseY - tile - row * pitch, w: tile, h: tile,
          fill: k < filled ? { color: C.amber } : { color: C.bg, transparency: 78 },
          line: { type: 'none' },
        });
      }
      K.text(slide, w.id, { x: x0 - 0.1, y: baseY + 0.08, w: blockW + 0.2, h: 0.28, fontSize: 14, bold: true, color: C.bg, align: 'center' });
      K.text(slide, `${w.beds} beds`, { x: x0 - 0.1, y: baseY + 0.36, w: blockW + 0.2, h: 0.26, fontSize: 12, color: HEX.lt2, align: 'center' });
    });
    K.text(slide, `Each square is one of the ${ctx.nBeds} funded beds of the simulated reference hospital (${ctx.nWards} wards, ${ctx.year0}–${ctx.year1}); the orange share is each ward's mean midnight occupancy.`, { x: 0.8, y: 6.84, w: 11.7, h: 0.3, fontSize: 12, color: HEX.lt2 });
    slide.addNotes(notesFor('A', 30, [
      `Good morning. BedSynth AI generates synthetic, non-clinical hospital operations data: SDV models learn bed-occupancy and scheduling data, and a FastAPI service returns validated records on demand.`,
      `Each square is one of the ${ctx.nBeds} funded beds of our simulated hospital; orange shows each ward's mean midnight occupancy.`,
      `I am Member A. Then Member B covers the SDV pipeline, Member C the evaluation, Member D the API and live demo, and Member C closes.`,
    ].join('\n\n')));
  }

  // ---------------------------------------------------------------- 2. Problem and goal
  {
    const slide = contentSlide({
      member: 'A', kicker: 'Opening · problem and goal', title: 'Optimisation models need realistic bed data', seconds: 35,
      notes: [
        `Bed managers and the analysts who build bed-allocation, elective-booking and rostering models need operational data to test with.`,
        `Three things stand in the way: ward and stay records are confidential facility data; the pressures a model must handle - winter surges, escalation, delayed discharges, short staffing - are rare in any one extract; and testing needs many datasets with known conditions.`,
        `Our goal, from the brief: generate realistic bed-occupancy and scheduling data with SDV, prove its quality and serve it through an API, in four weekly steps.`,
      ].join('\n\n'),
    });
    const rows = [
      ['lock', 'Real data is sensitive', 'Ward and patient-stay records are confidential facility data — they cannot be shared freely with developers or students.'],
      ['alert', 'Rare pressures are under-represented', 'Winter surges, escalation, delayed discharges and staffing shortages are exactly what a model must handle.'],
      ['engine', 'Testing needs data on demand', 'Bed allocation, elective booking and staffing models need many valid datasets with known conditions.'],
    ];
    rows.forEach(([icon, head, body], i) => {
      const y = TOP + 0.08 + i * 1.6;
      K.icon(slide, icon, MX, y, 0.62, i === 0 ? C.red : C.teal);
      K.text(slide, head, { x: MX + 0.85, y: y - 0.02, w: 4.9, h: 0.4, fontSize: 18, bold: true });
      K.text(slide, body, { x: MX + 0.85, y: y + 0.42, w: 4.9, h: 0.95, fontSize: 15, color: HEX.muted });
    });
    const gx = 6.85;
    const gw = W - MX - gx;
    K.box(slide, { x: gx, y: TOP, w: gw, h: 4.42, radius: 0.12, fill: C.tint, name: 'Goal card' });
    K.text(slide, 'OUR GOAL', { x: gx + 0.4, y: TOP + 0.32, w: 3, h: 0.3, fontSize: 13, bold: true, color: C.teal, charSpacing: 1.5 });
    K.text(slide, 'An SDV-based generator that produces realistic, operationally valid synthetic ward-day and stay data for any scenario in the brief — and measures its quality.', {
      x: gx + 0.4, y: TOP + 0.72, w: gw - 0.8, h: 1.75, fontSize: 21, color: C.ink, fontFace: HEAD_FONT,
    });
    const steps = [['Week 1', 'Data\npreparation'], ['Week 2', 'SDV\npipeline'], ['Week 3', 'Quality\nevaluation'], ['Week 4', 'FastAPI\ngenerator']];
    const lineY = TOP + 3.0;
    const x0 = gx + 0.95;
    const x1 = gx + gw - 0.95;
    K.line(slide, x0, lineY, x1, lineY, { color: HEX.accent3, width: 2 });
    steps.forEach(([wk, label], i) => {
      const cx = x0 + (i * (x1 - x0)) / (steps.length - 1);
      K.circle(slide, cx - 0.21, lineY - 0.21, 0.42, C.teal);
      K.text(slide, String(i + 1), { x: cx - 0.21, y: lineY - 0.21, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: C.bg, align: 'center', valign: 'middle' });
      K.text(slide, wk, { x: cx - 0.65, y: lineY + 0.33, w: 1.3, h: 0.3, fontSize: 14, bold: true, align: 'center' });
      K.text(slide, label, { x: cx - 0.65, y: lineY + 0.62, w: 1.3, h: 0.6, fontSize: 14, color: HEX.muted, align: 'center' });
    });
  }

  // ---------------------------------------------------------------- 3. Why synthetic data
  {
    const slide = contentSlide({
      member: 'A', kicker: 'Opening · motivation', title: 'Why synthetic data?', seconds: 20,
      notes: [
        `Why synthetic rather than anonymised data? Synthetic rows are sampled, not copied, and we measure copying. Testers can request exactly the conditions a model must handle, reproducibly from a seed. Every returned record passes all ${ctx.rules.ward_daily.hard} ward-day or ${ctx.rules.patient_stays.hard} stay hard rules. And every model is scored against a real-data ceiling and a no-learning floor.`,
      ].join('\n\n'),
    });
    const cards = [
      ['lock', 'Privacy checked', 'Synthetic rows are sampled, not copied; exact matches and the distance to the closest record are checked against real rows no model was trained on.'],
      ['engine', 'Testing optimisation models', 'Ask for the conditions a model must handle — high occupancy, emergency pressure, cancellations — in any volume, reproducible from a seed.'],
      ['check', 'Valid on delivery', `Constraints inside SDV plus rejection sampling: every returned record passes all ${ctx.rules.ward_daily.hard} ward-day or ${ctx.rules.patient_stays.hard} stay hard rules, so tests never get impossible data.`],
      ['chart', 'Measured, not assumed', 'Each model is scored against real holdout data (ceiling) and a no-learning baseline (floor) before it can become the default.'],
    ];
    const cw = (CW - 0.33) / 2;
    const ch = 2.3;
    cards.forEach(([icon, head, body], i) => {
      const x = MX + (i % 2) * (cw + 0.33);
      const y = TOP + Math.floor(i / 2) * (ch + 0.3);
      K.box(slide, { x, y, w: cw, h: ch, radius: 0.1, fill: C.tint, name: `Card ${head}` });
      K.icon(slide, icon, x + 0.32, y + 0.32, 0.62);
      K.text(slide, head, { x: x + 1.2, y: y + 0.3, w: cw - 1.5, h: 0.45, fontSize: 19, bold: true });
      K.text(slide, body, { x: x + 1.2, y: y + 0.82, w: cw - 1.5, h: 1.3, fontSize: 15, color: HEX.muted });
    });
  }

  // ---------------------------------------------------------------- 4. Architecture
  {
    const nEndpoints = ctx.endpoints.length;
    const slide = contentSlide({
      member: 'A', kicker: 'Opening · architecture', title: 'Architecture: from simulator to API', seconds: 35,
      notes: [
        `A discrete-event simulator produces the reference data of a ${ctx.nBeds}-bed, ${ctx.nWards}-ward hospital for ${ctx.year0} to ${ctx.year1} and injects data-quality problems. Week 1 cleans, repairs and validates it against ${ctx.hardRules} hard rules and splits it for SDV.`,
        `Week 2 trains ${word(nExperiments)} SDV experiments per table with ${word(nGroups)} constraint groups and registers each model with a model card. The generation engine turns a request into validated rows, and the FastAPI service exposes ${nEndpoints} endpoints, the demo UI among them. Underneath, Week 3 scores every model; the best composite per table becomes the API's default.`,
      ].join('\n\n'),
    });
    const stages = [
      ['hospital', 'Reference simulator', [`${ctx.nBeds} beds · ${ctx.nWards} wards`, `${ctx.year0}–${ctx.year1}`, '+ injected data issues'], 'data/raw/'],
      ['data', 'Week 1 · Preparation', ['clean · repair · validate', `${ctx.hardRules} hard rules`, 'train / holdout split'], 'data/processed/'],
      ['model', 'Week 2 · SDV models', [`${nExperiments} SDV experiments`, `${nGroups} constraint groups`, 'model cards + registry'], 'models/'],
      ['engine', 'Generation engine', ['conditions + filters', 'allocation strata', 'rejection sampling'], 'data/synthetic/'],
      ['api', 'Week 4 · FastAPI', [`${nEndpoints} endpoints + demo UI`, 'validated records', '+ metadata'], 'reports/api_demo/'],
    ];
    const bw = 2.17;
    const gap = (CW - stages.length * bw) / (stages.length - 1);
    const by = TOP + 0.1;
    const bh = 2.45;
    stages.forEach(([icon, head, lines, artefact], i) => {
      const x = MX + i * (bw + gap);
      K.box(slide, { x, y: by, w: bw, h: bh, radius: 0.1, fill: i === 2 || i === 3 ? C.tint : 'F4F8F8', line: HEX.rule, name: `Stage ${head}` });
      K.icon(slide, icon, x + 0.18, by + 0.2, 0.52);
      K.text(slide, head, { x: x + 0.18, y: by + 0.84, w: bw - 0.3, h: 0.55, fontSize: 15, bold: true });
      K.text(slide, lines.join('\n'), { x: x + 0.18, y: by + 1.42, w: bw - 0.26, h: 0.95, fontSize: 14, color: HEX.muted, paraSpaceAfter: 2 });
      K.text(slide, artefact, { x, y: by + bh + 0.1, w: bw, h: 0.3, fontSize: 12, fontFace: CODE_FONT, color: HEX.muted, align: 'center' });
      if (i < stages.length - 1) K.line(slide, x + bw + 0.06, by + bh / 2, x + bw + gap - 0.06, by + bh / 2, { color: HEX.accent1, width: 2, arrow: true });
    });
    // evaluation (Week 3) under the SDV models and the engine
    const ex = MX + 2 * (bw + gap);
    const ew = 2 * bw + gap;
    const ey = by + bh + 0.8;
    const eh = 1.3;
    K.box(slide, { x: ex, y: ey, w: ew, h: eh, radius: 0.1, fill: C.ink, name: 'Evaluation' });
    K.icon(slide, 'chart', ex + 0.22, ey + (eh - 0.56) / 2, 0.56, C.bg, C.ink);
    K.text(slide, [
      { text: 'Week 3 · Evaluation', options: { bold: true, color: C.bg, breakLine: true } },
      { text: `${families.length} composite components vs real data; real holdout = ceiling, no-learning baseline = floor; best composite → registry default`, options: { color: HEX.lt2 } },
    ], { x: ex + 0.98, y: ey + 0.08, w: ew - 1.18, h: eh - 0.16, fontSize: 14, valign: 'middle' });
    K.line(slide, ex + bw / 2, ey - 0.03, ex + bw / 2, by + bh + 0.46, { color: HEX.dk2, width: 2, arrow: true });
    K.text(slide, 'selects the default model', { x: ex + bw / 2 + 0.12, y: by + bh + 0.47, w: 2.6, h: 0.28, fontSize: 12, italic: true, color: HEX.muted });
  }

  // ---------------------------------------------------------------- 5. The data
  {
    const sz = ctx.season;
    const wardName = (id) => (ctx.wards.find((w) => w.id === id) || {}).name || id;
    const seasonText = sz.winterPeakSummerLow ? 'winter peaks and a summer dip' : `a peak in ${MONTH_NAMES[sz.peak - 1]} and a low in ${MONTH_NAMES[sz.low - 1]}`;
    const slide = contentSlide({
      member: 'A', kicker: 'The data', title: 'Two tables from a simulated hospital', seconds: 55,
      source: 'Chart: mean midnight occupancy per ward and month in data/processed/ward_daily_clean.csv (reference data)',
      notes: [
        `No real patient-level data can be shared, so the reference data comes from our own simulator: ${ctx.nBeds} funded beds in ${ctx.nWards} wards over ${ctx.nYears} years. The schema in bedsynth/schema.py is the source of truth, so a real extract with the same columns could replace it.`,
        `ward_daily has one row per ward per day: ${fmtInt(wd.finalRows)} clean ward-days with capacity, staffing, census, electives, emergency flow, discharges, delays and boarders. patient_stays has one row per ward stay: ${fmtInt(ps.finalRows)} clean stays with admission, booking, bed wait, length of stay and discharge details.`,
        `SDV learns only ${ctx.cols.ward_daily.training} and ${ctx.cols.patient_stays.training} of the columns; derived ones such as the occupancy rate are recomputed. The chart shows the seasonal pattern to reproduce: ${seasonText}, with the largest swings on the ${listJoin(sz.strongest)} wards (${listJoin(sz.strongest.map(wardName))}).`,
      ].join('\n\n'),
    });
    const stats = [
      [String(ctx.nBeds), 'funded beds'],
      [String(ctx.nWards), 'wards'],
      [String(ctx.nYears), `years (${ctx.year0}–${ctx.year1})`],
      [fmtInt(wd.finalRows), 'ward-days'],
      [fmtInt(ps.finalRows), 'patient stays'],
    ];
    const sw = (CW - 4 * 0.25) / 5;
    stats.forEach(([v, l], i) => K.stat(slide, MX + i * (sw + 0.25), TOP - 0.05, sw, v, l));
    const cy = TOP + 1.3;
    const cardW = 6.05;
    const tablesInfo = [
      ['ward_daily', 'one row per ward per day', 'escalation and closed beds · nurse staffing · census · electives and cancellations · emergency admissions · transfers · discharges and delays · ED boarders · waits · occupancy band'],
      ['patient_stays', 'one row per ward stay', 'admission source and timing · booking lead and reschedules · bed wait · length of stay · discharge timing and destination · delay reason and hours · bed turnaround'],
    ].map(([t, grain, attrs]) => [t, grain, attrs.replace(/ · /g, `${NBSP}· `)]);
    tablesInfo.forEach(([t, grain, attrs], i) => {
      const y = cy + i * 1.86;
      K.box(slide, { x: MX, y, w: cardW, h: 1.74, radius: 0.1, fill: C.tint, name: `Table ${t}` });
      K.text(slide, [
        { text: t, options: { fontFace: CODE_FONT, bold: true, color: C.ink } },
        { text: `  ·  ${grain}`, options: { bold: true } },
      ], { x: MX + 0.25, y: y + 0.16, w: cardW - 0.5, h: 0.34, fontSize: 16 });
      K.text(slide, `${ctx.cols[t].training} learned columns + ${ctx.cols[t].derived} derived · ${fmtInt(ctx.dq[t].finalRows)} rows`, { x: MX + 0.25, y: y + 0.5, w: cardW - 0.5, h: 0.3, fontSize: 14, color: C.teal, bold: true });
      K.text(slide, attrs, { x: MX + 0.25, y: y + 0.84, w: cardW - 0.45, h: 0.8, fontSize: 14, color: HEX.muted });
    });
    // native line chart (replaces the week-1 PNG, whose labels were about 6 pt at slide size)
    const fw = 5.7;
    const wardsShown = ctx.wards.filter((w) => sz.byWard[w.id]);
    const lo = Math.floor(sz.min * 10) / 10;
    const hi = Math.max(lo + 0.1, Math.ceil(sz.max * 10) / 10);
    slide.addChart(pres.ChartType.line, wardsShown.map((w) => ({ name: w.id, labels: sz.months.map((m) => MONTHS[m - 1]), values: sz.byWard[w.id] })), Object.assign(chartBase(), {
      x: W - MX - fw, y: cy - 0.04, w: fw, h: 3.66, chartColors: WARD_COLORS.slice(0, wardsShown.length), lineSize: 2, lineDataSymbol: 'none',
      showLegend: true, legendPos: 'b', legendFontSize: 11, showTitle: true, title: 'Mean midnight occupancy by month and ward', titleFontSize: 13,
      valAxisMinVal: lo, valAxisMaxVal: hi, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: '0%', catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
      altText: `Line chart of the reference data: mean midnight occupancy per calendar month for each of the ${word(wardsShown.length)} wards.`,
    }));
  }

  // ---------------------------------------------------------------- 6. Week 1 preparation
  {
    // injected issue types, read from the injection logs (types of one table only are named as such)
    const issueName = {
      alt_date_format: 'alternative date formats', category_variant: 'spelling variants', duplicate_row: 'duplicates',
      invalid_value: 'invalid values', missing_value: 'missing values', timestamp_swap: 'swapped timestamps',
    };
    const typesOf = (q) => Object.keys(q.injectedTypes).filter((k) => q.injectedTypes[k] > 0);
    const sharedTypes = typesOf(wd).filter((k) => typesOf(ps).includes(k));
    const onlyOne = TABLES.map((t) => [t, typesOf(ctx.dq[t]).filter((k) => !sharedTypes.includes(k))]).filter(([, ks]) => ks.length);
    const issueText = [listJoin(sharedTypes.map((k) => issueName[k] || human(k))),
      ...onlyOne.map(([t, ks]) => `for ${t === 'patient_stays' ? 'stays' : 'ward-days'} ${listJoin(ks.map((k) => issueName[k] || human(k)))}`)].filter(Boolean).join(', and ');
    const validFinal = wd.validFinal === ps.validFinal ? pct(wd.validFinal) : `${pct(wd.validFinal, 1)} and ${pct(ps.validFinal, 1)}`;
    const slide = contentSlide({
      member: 'A', kicker: 'Week 1 · data preparation', title: 'Cleaning a deliberately dirty extract', seconds: 65,
      source: 'Sources: data/processed/data_quality_report.json (issues per type, rates, split), schema_catalog.json (rules, columns)',
      notes: [
        `Week 1 produces the SDV-ready dataset. The simulator injects typical extract problems - ${issueText} - and logs each one, so cleaning can be measured.`,
        `The charts compare injected and detected issues per type${wd.everyTypeFound && ps.everyTypeFound ? '; every type is found at least as often as it was injected' : ''}. Of ${fmtInt(wd.injected)} issues injected into ward_daily and ${fmtInt(ps.injected)} into patient_stays, ${wd.resolved === 1 && ps.resolved === 1 ? 'all were resolved' : `${pct(wd.resolved, 1)} and ${pct(ps.resolved, 1)} were resolved`}, and ${pct(wd.exact, 1)} and ${pct(ps.exact, 1)} were restored to the exact original value. Cell by cell, the cleaned tables agree with the simulator's ground truth on ${pct(wd.cellAgreement, 1)} and ${pct(ps.cellAgreement, 1)} of cells.`,
        `Hard-rule validity rises from ${pct(wd.validRaw, 1)} and ${pct(ps.validRaw, 1)} to ${validFinal}; removing duplicates keeps ${pct(wd.retention, 1)} and ${pct(ps.retention, 1)} of rows. The data is split ${pct(1 - wd.holdoutFraction)} / ${pct(wd.holdoutFraction)} into training and holdout, stratified by ${human(wd.stratifyBy).replace(' id', '')}, and the SDV metadata covers only the learned columns.`,
      ].join('\n\n'),
    });
    // native charts of injected vs detected issues per type (replace the week-1 PNG, whose labels were about 4 pt at slide size)
    const fw = 6.35;
    const fh = 2.5;
    const cgap = 0.2;
    TABLES.forEach((t, i) => {
      const q = ctx.dq[t];
      // reversed so the alphabetical order reads top-down with the value axis in its default place (bottom)
      const kinds = [...new Set([...Object.keys(q.injectedTypes), ...Object.keys(q.detectedTypes)])].sort().reverse();
      const cx = MX + i * ((fw - cgap) / 2 + cgap);
      if (!kinds.length) {
        K.text(slide, `${t}: no injected or detected issues logged`, { x: cx, y: TOP + 0.2, w: (fw - cgap) / 2, h: 0.6, fontSize: 14, color: HEX.muted });
        return;
      }
      const top = Math.max(1, ...kinds.map((k) => Math.max(q.injectedTypes[k] || 0, q.detectedTypes[k] || 0)));
      const step = niceStep(top);
      slide.addChart(pres.ChartType.bar, [
        { name: 'Injected', labels: kinds.map(human), values: kinds.map((k) => q.injectedTypes[k] || 0) },
        { name: 'Detected', labels: kinds.map(human), values: kinds.map((k) => q.detectedTypes[k] || 0) },
      ], Object.assign(chartBase(), {
        x: cx, y: TOP - 0.04, w: (fw - cgap) / 2, h: fh + 0.04, barDir: 'bar', barGrouping: 'clustered',
        chartColors: [HEX.accent4, HEX.accent1], barGapWidthPct: 45,
        valAxisMinVal: 0, valAxisMaxVal: Math.ceil(top / step) * step, valAxisMajorUnit: step, valAxisLabelFormatCode: '#,##0',
        catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, legendFontSize: 11, showLegend: true, legendPos: 't',
        showTitle: true, title: t, titleFontSize: 13, showValue: false,
        altText: `Bar chart: injected versus detected data-quality issues per issue type in ${t}.`,
      }));
    });
    const steps = ['Standardise', 'De-duplicate', 'Repair', 'Validate & split'];
    const chevY = TOP + fh + 0.38;
    const chevW = (fw + 0.12 * (steps.length - 1)) / steps.length;
    steps.forEach((s, i) => {
      const x = MX + i * (chevW - 0.12);
      slide.addShape(i === 0 ? pres.ShapeType.homePlate : pres.ShapeType.chevron, {
        x, y: chevY, w: chevW, h: 0.56, fill: { color: i === steps.length - 1 ? C.teal : C.tint }, line: { type: 'none' },
      });
      K.text(slide, s, { x: x + (i === 0 ? 0.12 : 0.3), y: chevY, w: chevW - 0.5, h: 0.56, fontSize: 14, bold: true, align: 'center', valign: 'middle', color: i === steps.length - 1 ? C.bg : C.ink });
    });
    K.text(slide, `Checked cell by cell against the simulator's ground truth, the cleaned tables agree on ${pct(wd.cellAgreement, 1)} (ward_daily) and ${pct(ps.cellAgreement, 1)} (patient_stays) of all cells.`, {
      x: MX, y: chevY + 0.78, w: fw, h: 0.62, fontSize: 14, color: HEX.muted,
    });
    const tx = 7.2;
    const tw = W - MX - tx;
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.bg, fill: { color: C.ink }, fontFace: CODE_FONT, align: 'center' } });
    const row = (label, a, b, i) => [
      { text: label, options: { bold: true, fill: { color: i % 2 ? 'FFFFFF' : HEX.lt2 } } },
      { text: a, options: { align: 'center', fill: { color: i % 2 ? 'FFFFFF' : HEX.lt2 } } },
      { text: b, options: { align: 'center', fill: { color: i % 2 ? 'FFFFFF' : HEX.lt2 } } },
    ];
    const both = (fn) => [fn(wd, 'ward_daily'), fn(ps, 'patient_stays')];
    const rowsData = [
      ['Raw rows', ...both((x) => fmtInt(x.rawRows))],
      ['Issues injected → found', ...both((x) => `${fmtInt(x.injected)} → ${fmtInt(x.detected)}`)],
      ['Injected fixed (exact)', ...both((x) => `${pct(x.resolved)} (${pct(x.exact, 1)})`)],
      ['Hard-valid: raw → clean', ...both((x) => `${pct(x.validRaw, 1)} → ${pct(x.validFinal)}`)],
      ['Rows kept', ...both((x) => `${fmtInt(x.finalRows)} (${pct(x.retention, 1)})`)],
      ['Train / holdout', ...both((x) => `${fmtInt(x.train)} / ${fmtInt(x.holdout)}`)],
      ['Hard / soft rules', ...TABLES.map((t) => `${ctx.rules[t].hard} / ${ctx.rules[t].soft}`)],
      ['Columns learned by SDV', ...TABLES.map((t) => `${ctx.cols[t].training} of ${ctx.cols[t].total}`)],
    ];
    const rows = [[{ text: '', options: { fill: { color: C.ink } } }, hdr('ward_daily'), hdr('patient_stays')]]
      .concat(rowsData.map((r, i) => row(r[0], r[1], r[2], i)));
    slide.addTable(rows, {
      x: tx, y: TOP, w: tw, colW: [2.26, (tw - 2.26) / 2, (tw - 2.26) / 2], fontSize: 14, color: HEX.dk1,
      border: { type: 'solid', pt: 0.75, color: HEX.rule }, rowH: 0.43, valign: 'middle', margin: [0.03, 0.08, 0.03, 0.08],
    });
  }

  // ---------------------------------------------------------------- 7. Scenarios (Week 1 catalogue and reference support)
  {
    const familyOrder = Object.keys(ctx.categories).filter((c) => c !== 'baseline');
    const baselines = ctx.scenarios.filter((s) => s.category === 'baseline').map((s) => s.name);
    const byTable = (t, c) => ctx.scenarios.filter((s) => s.table === t && s.category === c).map((s) => s.title);
    const counts = Object.fromEntries(TABLES.map((t) => [t, ctx.scenarios.filter((s) => s.table === t).length]));
    const allCovered = ctx.familiesCovered.length === ctx.nFamilies;
    const ex = ctx.scenarios.find((s) => Object.keys(s.conditions || {}).length && Object.keys(s.filters || {}).length && s.table === 'ward_daily') || ctx.scenarios[0];
    const sup = ctx.support;
    // brief scenario families with their reference support are part of the Week 1 data preparation
    // (Week 1 report section 1.2; docs/TEAM_PLAN.md presentation plan, part 2)
    const slide = contentSlide({
      member: 'A', kicker: 'Week 1 · scenario catalogue', seconds: 60,
      title: allCovered ? `${ctx.nScenarios} scenarios cover every brief family` : `${ctx.nScenarios} scenarios across ${ctx.familiesCovered.length} brief families`,
      source: 'Source: scenario catalogue (bedsynth/scenarios.py) with reference support, as served by GET /api/v1/scenarios in reports/api_demo',
      notes: [
        `The brief lists the scenario families the generator must cover. Our catalogue has ${ctx.nScenarios} named scenarios - ${counts.ward_daily} for ward_daily and ${counts.patient_stays} for patient_stays - and each table row is one brief family, so ${allCovered ? `all ${ctx.nFamilies} families are covered` : `${ctx.familiesCovered.length} of ${ctx.nFamilies} families are covered`}, plus an unconditional baseline per table.`,
        `A scenario is just data: conditions - values of categorical training columns such as the occupancy band - and filters, bounds on any column, derived ones included. ${ex.name}, for example, combines ${condText(ex.conditions) || 'no condition'} with ${filterText(ex.filters) || 'no filter'}.`,
        `${sup ? `Every scenario has reference rows to learn from: between ${fmtInt(sup.min.support.matching_rows)} (${sup.min.name}) and ${fmtInt(sup.max.support.matching_rows)} (${sup.max.name}) in the ${sup.split === 'train' ? 'training' : sup.split} split. ` : ''}Member B shows how the engine meets the conditions, and Member C how well each scenario is reproduced.`,
      ].join('\n\n'),
    });
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.bg, fill: { color: C.ink } } });
    const rows = [[hdr('Brief scenario family'), hdr(`ward_daily (${counts.ward_daily})`), hdr(`patient_stays (${counts.patient_stays})`)]];
    familyOrder.forEach((c, i) => {
      const fill = { color: i % 2 ? 'FFFFFF' : HEX.lt2 };
      const a = byTable('ward_daily', c);
      const b = byTable('patient_stays', c);
      rows.push([
        { text: ctx.categories[c] || human(c), options: { bold: true, fill } },
        { text: a.length ? a.join(' · ') : '–', options: { fill } },
        { text: b.length ? b.join(' · ') : '–', options: { fill } },
      ]);
    });
    slide.addTable(rows, {
      x: MX, y: TOP - 0.02, w: CW, colW: [3.55, 4.85, 3.73], fontSize: 14, color: HEX.dk1,
      border: { type: 'solid', pt: 0.75, color: HEX.rule }, valign: 'middle', margin: [0.025, 0.08, 0.025, 0.08], rowH: 0.36,
    });
    K.box(slide, { x: MX, y: 5.78, w: CW, h: 0.84, radius: 0.08, fill: C.tint, name: 'Scenario definition' });
    K.text(slide, [
      { text: 'A scenario = conditions + filters. ', options: { bold: true } },
      { text: `${ex.name}: `, options: { fontFace: CODE_FONT, color: C.ink } },
      { text: `condition ${condText(ex.conditions)} (a categorical training column); filter ${filterText(ex.filters)} (any column, checked after derivation).`, options: { breakLine: true } },
      {
        text: sup
          ? `Reference support: ${fmtInt(sup.min.support.matching_rows)}–${fmtInt(sup.max.support.matching_rows)} matching ${sup.split === 'train' ? 'training' : sup.split} rows per scenario (fewest: ${human(sup.min.name)}); one baseline per table.`
          : `Plus one unconditional baseline per table (${baselines.join(', ')}).`,
        options: { color: HEX.muted },
      },
    ], { x: MX + 0.22, y: 5.78, w: CW - 0.44, h: 0.84, fontSize: 14, valign: 'middle' });
  }

  // ---------------------------------------------------------------- 8. Week 2 SDV pipeline
  {
    const slide = contentSlide({
      member: 'B', kicker: 'Week 2 · SDV pipeline', title: `${cap(word(nExperiments))} SDV experiments per table`, seconds: 85,
      // the notes also answer "why not HMA or PAR?" (single-table design, docs/DESIGN.md section 13)
      source: 'Sources: models/registry.json and model cards',
      footnote: quick,
      notes: [
        `Thank you. I am Member B. Training follows the steps along the top: metadata from our schema, optional segmentation, constraints, one SDV synthesizer per segment, and a model card in the registry.`,
        `We train ${word(nExperiments)} experiments per table. The first ${word(ABLATION.length)} are an ablation of the GaussianCopula recipe: ${ABLATION[0]} is one global model without constraints, ${ABLATION[1]} adds segments, SDV's built-in constraints and tuned marginals, and ${ABLATION[2]} adds our programmable constraints. CTGAN and TVAE are the deep-learning alternatives${ctx.epochs.length ? `, trained for ${ctx.epochs.join('/')} epochs` : ''}. Segments are per ward for ward_daily (${d.cards.ward_daily[ABLATION[2]].n_segments}) and per ward and admission source for stays (${d.cards.patient_stays[ABLATION[2]].n_segments}), because one copula cannot capture how different intensive care is from a surgical ward.`,
        `Of the ${word(nGroups)} constraint groups, ${listJoin(builtinGroups.map(human))} ${builtinGroups.length === 1 ? 'is an SDV built-in' : 'are SDV built-ins'}; the other ${word(progGroups.length)} are our own programmable classes. The Week 3 evaluation picks the default: ${sameDefault ? `${dWD} for both tables` : `${dWD} for ward_daily and ${dPS} for patient_stays`}.`,
        `Why two single-table models? SDV's multi-table HMASynthesizer needs single-column keys, but our tables link only through the composite ward × date, and one stay spans several ward-days. The sequential PARSynthesizer would add day-to-day continuity, but our scenarios condition and validate individual rows, so continuity stays future work.`,
      ].join('\n\n'),
    });
    const steps = [['Metadata', 'from schema.py'], ['Segment', 'ward or ward × source'], ['Constrain', `${nGroups} groups`], ['Fit', 'one per segment'], ['Register', 'model card + registry']];
    const chevW = (CW + 0.16 * (steps.length - 1)) / steps.length;
    steps.forEach(([a, b], i) => {
      const x = MX + i * (chevW - 0.16);
      slide.addShape(i === 0 ? pres.ShapeType.homePlate : pres.ShapeType.chevron, { x, y: TOP, w: chevW, h: 0.78, fill: { color: i % 2 ? C.tint : 'DCEBE9' }, line: { type: 'none' } });
      K.text(slide, [{ text: a, options: { bold: true, breakLine: true, color: C.ink } }, { text: b, options: { color: HEX.muted } }], {
        x: x + (i === 0 ? 0.18 : 0.42), y: TOP, w: chevW - 0.72, h: 0.78, fontSize: 14, align: 'center', valign: 'middle',
      });
    });
    const groupsText = (card) => {
      const gs = card.constraint_groups || [];
      if (!gs.length) return 'none';
      if (gs.length === nGroups) return `all ${nGroups}`;
      if (gs.every((g) => ctx.groupKind[g] !== 'programmable')) return `${gs.length} built-in`;
      return `${gs.length} of ${nGroups}`;
    };
    const synthName = (card) => String(card.synthesizer || card.model || '').replace(/Synthesizer$/, '');
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.bg, fill: { color: C.ink } } });
    const rows = [[hdr('Experiment'), hdr('Synthesizer'), hdr('Segments: ward_daily / patient_stays'), hdr('Constraint groups'), hdr('Epochs'), hdr('Fit time (s)')]];
    ctx.experiments.forEach((id, i) => {
      const a = d.cards.ward_daily[id];
      const b = d.cards.patient_stays[id];
      const any = a || b;
      const isDefault = TABLES.filter((t) => ctx.defaults[t] === id);
      const fill = { color: isDefault.length ? 'D6EAE7' : i % 2 ? 'FFFFFF' : HEX.lt2 };
      const label = isDefault.length === TABLES.length ? `${id} (default)` : isDefault.length ? `${id} (default: ${isDefault.join(', ')})` : id;
      const ep = opt(any, 'training.sdv_params.epochs');
      rows.push([
        { text: label, options: { bold: true, fontFace: CODE_FONT, fill } },
        { text: synthName(any), options: { fill } },
        { text: `${a ? segText(a) : '–'} / ${b ? segText(b) : '–'}`, options: { fill } },
        { text: groupsText(any), options: { fill, align: 'center' } },
        { text: ep === null ? '–' : String(ep), options: { fill, align: 'center' } },
        { text: `${a ? Math.round(a.fit_seconds) : '–'} / ${b ? Math.round(b.fit_seconds) : '–'}`, options: { fill, align: 'center' } },
      ]);
    });
    slide.addTable(rows, {
      x: MX, y: TOP + 1.05, w: CW, colW: [2.75, 1.85, 3.43, 1.75, 0.95, 1.4], fontSize: 14, color: HEX.dk1,
      border: { type: 'solid', pt: 0.75, color: HEX.rule }, rowH: 0.42, valign: 'middle', margin: [0.03, 0.08, 0.03, 0.08],
    });
    const chipRow = (y, label, groups, fill, color) => {
      K.text(slide, label, { x: MX, y, w: 2.75, h: 0.4, fontSize: 14, bold: true, valign: 'middle' });
      let x = MX + 2.85;
      groups.forEach((g) => {
        const w = 0.42 + human(g).length * 0.098;
        K.box(slide, { x, y: y + 0.02, w, h: 0.36, radius: 0.18, fill });
        K.text(slide, human(g), { x, y: y + 0.02, w, h: 0.36, fontSize: 14, color, align: 'center', valign: 'middle' });
        x += w + 0.16;
      });
    };
    const chipsY = TOP + 1.05 + 0.42 * (rows.length) + 0.32;
    chipRow(chipsY, 'SDV built-in', builtinGroups, 'DCEBE9', C.ink);
    chipRow(chipsY + 0.5, 'BedSynth programmable', progGroups, C.teal, C.bg);
  }

  // ---------------------------------------------------------------- 9. Programmable constraints
  {
    const record = opt(d.demoGenerate.evidence, 'response.body.records.0');
    if (!record) fail('API demo generation evidence has no records');
    const census = num(record.census_end, 'evidence record census_end');
    const avail = num(record.available_beds, 'evidence record available_beds');
    const rate = census / avail;
    let band = ctx.bandNames[ctx.bandNames.length - 1];
    for (let i = 0; i < ctx.bandThresholds.length; i += 1) {
      if (rate < ctx.bandThresholds[i]) { band = ctx.bandNames[i]; break; }
    }
    if (band !== record.occupancy_level) console.warn(`warning: recomputed band ${band} differs from the record label ${record.occupancy_level}`);
    const bandLine = ctx.bandNames.map((b, i) => (i < ctx.bandThresholds.length ? `${b} < ${f2(ctx.bandThresholds[i])} ≤ ` : b)).join('');
    const wdValidity = pick(resWD, 'validity.hard_violation_rates', `ward_daily/${dWD} results`);
    const wdH10 = opt(wdValidity, 'WD-H10');
    const bandExpr = (d.catalog.tables.ward_daily.constraints.find((c) => c.group === 'regime_labels' && (c.columns || []).includes('occupancy_level')) || {}).expression;
    if (!bandExpr) fail('schema_catalog.json: no regime_labels constraint for occupancy_level');
    const slide = contentSlide({
      member: 'B', kicker: 'Week 2 · constraints', title: 'How a programmable constraint works', seconds: 75,
      source: 'Sources: data/processed/schema_catalog.json, bedsynth/synthesis/constraints.py; example record from reports/api_demo',
      footnote: quick,
      notes: [
        `SDV${ctx.sdvVersion ? ` ${ctx.sdvVersion}` : ''} lets us write constraints by subclassing SingleTableProgrammableConstraint: transform runs on the training data, reverse_transform on every sampled row.`,
        `Take the occupancy band. Its transform is the identity, so the band stays in the model as a latent regime variable that carries its correlation with demand, staffing and boarding. Its reverse_transform recomputes the band from the sampled counts with the schema's own function, so the band can never contradict the census: rule WD-H10 holds by construction.`,
        `In a row from the API demo, a midnight census of ${census} on ${avail} available beds gives ${f3(rate)}: the ${band} band, whatever label the copula sampled. In raw ${dWD} samples, WD-H10 is broken in ${wdH10 === null ? 'an unreported share' : pct(wdH10, 1)} of rows.`,
        `The same pattern gives ${word(progGroups.length)} programmable groups: ${listJoin(progGroups.map((g) => ({ regime_labels: 'band labels', structural_zeros: 'learned structural zeros such as no electives on Sundays', capacity: 'the escalation-bed cover', conditional_marginals: 'exact per-group marginals' }[g] || human(g))))}. Order matters: SDV reverses transforms in reverse order, so our constraints are added first and their repair runs last, on fully restored columns.`,
      ].join('\n\n'),
    });
    const nodes = [
      ['Training rows', 'real ward-days, one model per ward', false],
      ['transform()', 'identity: the band stays in the model as a latent regime variable', true],
      ['SDV fit + sample', `GaussianCopula per ward (${d.cards.ward_daily[ABLATION[2]].n_segments}${NBSP}segments)`, false],
      ['reverse_transform()', 'band recomputed from the sampled counts, so WD-H10 holds by construction', true],
    ];
    const ngap = 0.42;
    const nw = (CW - (nodes.length - 1) * ngap) / nodes.length;
    nodes.forEach(([head, body, hot], i) => {
      const x = MX + i * (nw + ngap);
      K.box(slide, { x, y: TOP, w: nw, h: 1.32, radius: 0.1, fill: hot ? C.teal : C.tint, name: `Step ${head}` });
      K.text(slide, [
        { text: head, options: { bold: true, breakLine: true, fontFace: /\(\)/.test(head) ? CODE_FONT : undefined, color: hot ? C.bg : C.ink } },
        { text: body, options: { color: hot ? C.bg : HEX.muted } },
      ], { x: x + 0.14, y: TOP + 0.1, w: nw - 0.28, h: 1.12, fontSize: 14, valign: 'middle', align: 'center' });
      if (i < nodes.length - 1) K.line(slide, x + nw + 0.04, TOP + 0.66, x + nw + ngap - 0.04, TOP + 0.66, { color: HEX.accent1, width: 2, arrow: true });
    });
    // example card
    const ey = TOP + 1.62;
    const ew = 6.3;
    K.box(slide, { x: MX, y: ey, w: ew, h: 3.3, radius: 0.1, fill: 'F4F8F8', line: HEX.rule, name: 'Example' });
    K.text(slide, 'Example: the occupancy band (rule WD-H10)', { x: MX + 0.25, y: ey + 0.16, w: ew - 0.5, h: 0.34, fontSize: 16, bold: true });
    const [lhs, rhs] = bandExpr.split(/\s*=\s*/);
    K.text(slide, `${lhs} =\n    ${rhs}`, { x: MX + 0.25, y: ey + 0.58, w: ew - 0.5, h: 0.62, fontSize: 14, fontFace: CODE_FONT, color: C.ink });
    K.text(slide, bandLine, { x: MX + 0.25, y: ey + 1.28, w: ew - 0.5, h: 0.3, fontSize: 14, color: HEX.muted });
    K.text(slide, [
      { text: 'Generated ward-day (API demo): ', options: { bold: true } },
      { text: `census_end ${census}, available_beds ${avail} → ${census} / ${avail} = ${f3(rate)} → `, options: {} },
      { text: band, options: { bold: true, color: C.teal } },
    ], { x: MX + 0.25, y: ey + 1.68, w: ew - 0.5, h: 0.62, fontSize: 14 });
    K.text(slide, `Any other sampled label is overwritten. Raw ${dWD} samples: WD-H10 violation rate ${wdH10 === null ? 'n/a' : pct(wdH10, 1)}.`, { x: MX + 0.25, y: ey + 2.42, w: ew - 0.5, h: 0.7, fontSize: 14, color: HEX.muted });
    // four programmable groups
    const lx = MX + ew + 0.35;
    const lw = W - MX - lx;
    K.text(slide, `All ${word(progGroups.length)} programmable groups`, { x: lx, y: ey + 0.02, w: lw, h: 0.34, fontSize: 16, bold: true });
    const what = {
      regime_labels: 'bands recomputed from the counts',
      structural_zeros: 'e.g. no electives on Sundays',
      capacity: 'escalation = census cover + learned extra',
      conditional_marginals: 'exact per-group marginals',
    };
    progGroups.forEach((g, i) => {
      const y = ey + 0.5 + i * 0.7;
      const rules = ctx.groupRules[g] || [];
      K.circle(slide, lx, y + 0.07, 0.2, C.teal);
      K.text(slide, [
        { text: `${cap(human(g))}: `, options: { bold: true } },
        { text: what[g] || 'programmable constraint', options: { breakLine: true } },
        { text: rules.length ? `enforces ${rules.join(', ')}` : 'shapes the learned distribution', options: { color: HEX.muted } },
      ], { x: lx + 0.32, y, w: lw - 0.32, h: 0.64, fontSize: 14 });
    });
  }

  // ---------------------------------------------------------------- 10. Generation engine
  {
    const body = pick(d.demoGenerate.evidence, 'response.body', 'generation evidence');
    const req = pick(d.demoGenerate.evidence, 'request.json', 'generation evidence');
    const g = pick(body, 'generation', 'generation evidence');
    const v = pick(body, 'validation', 'generation evidence');
    const strata = opt(g, 'allocation_strata', {});
    const rejected = Object.entries(opt(g, 'rejected_by_rule', {}));
    const calBody = pick(d.demoCalendar.evidence, 'response.body', 'calendar evidence');
    const calReq = pick(d.demoCalendar.evidence, 'request.json', 'calendar evidence');
    const calDays = Math.round((Date.parse(calReq.end_date) - Date.parse(calReq.start_date)) / 86400000) + 1;
    const calFull = calBody.n_rows === (calReq.wards || []).length * calDays;
    const slide = contentSlide({
      member: 'B', kicker: 'Week 2 · generation engine', title: 'Generation engine: valid rows, right mix', seconds: 80,
      source: `Example: API demo step ${d.demoGenerate.step.index} (reports/api_demo/${d.demoGenerate.step.evidence})`,
      notes: [
        `Every row leaves the system through the generation engine: the API, the scripts and the evaluation all call it.`,
        `Step one merges the request: scenario conditions, request conditions and wards are intersected and filters tightened, so a request never leaves its scenario. Step two plans the work: each condition cell is split into allocation strata - ward by occupancy band, or ward by source by delay reason for stays - with quotas from the reference mix, because rejection accepts some strata far more easily than others. Filtered requests take their strata from a synthetic pilot draw, never from the matching real rows; calendar mode plans a ward-by-date grid.`,
        `Step three samples and rejects: conditions select the matching segment models and are met exactly by rejection from pooled draws, with SDV's native conditional sampling only for combinations too rare to pool. Rows that break a hard rule, a condition or a filter are dropped, and batch sizes follow the acceptance. Step four assembles the output and its report.`,
        `The demo call on the right asked for ${req.num_rows} ${req.scenario} rows: ${fmtInt(g.sampled_rows)} sampled, ${fmtInt(g.accepted_rows)} passed (${pct(g.acceptance_rate, 1)}), ${fmtInt(g.num_rows_generated)} kept by the strata quotas, in ${f2(g.elapsed_seconds)} of its ${Math.round(g.time_budget_seconds)} seconds${v.all_hard_rules_passed ? ', with every hard rule passed' : ''}.`,
      ].join('\n\n'),
    });
    const steps = [
      ['Merge the request', 'scenario conditions ∩ request conditions ∩ wards; filters tightened to the strictest bounds'],
      ['Plan cells and strata', 'condition cells split into allocation strata with reference quotas; calendar mode: wards × dates grid'],
      ['Sample, derive, reject — repeat', 'SDV sampling per cell, derived columns, reject hard-rule / condition / filter failures; adaptive batches until the quota or the time budget'],
      ['Assemble and report', 'seeded shuffle or ward/date order; report with seed, acceptance, rejections per rule, strata and warnings'],
    ];
    const lw = 6.75;
    steps.forEach(([head, desc], i) => {
      const y = TOP + i * 1.2;
      K.circle(slide, MX, y + 0.05, 0.5, i === 2 ? C.amber : C.teal);
      K.text(slide, String(i + 1), { x: MX, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.bg, align: 'center', valign: 'middle' });
      K.text(slide, head, { x: MX + 0.72, y, w: lw - 0.72, h: 0.36, fontSize: 17, bold: true });
      K.text(slide, desc, { x: MX + 0.72, y: y + 0.38, w: lw - 0.72, h: 0.72, fontSize: 14, color: HEX.muted });
      if (i < steps.length - 1) K.line(slide, MX + 0.25, y + 0.6, MX + 0.25, y + 1.2, { color: HEX.rule, width: 2 });
    });
    slide.addShape(pres.ShapeType.circularArrow, { x: MX + lw - 0.5, y: TOP + 2.4 - 0.1, w: 0.42, h: 0.42, fill: { color: C.amber }, line: { type: 'none' }, objectName: 'Loop' });
    const px = MX + lw + 0.3;
    const pw = W - MX - px;
    K.box(slide, { x: px, y: TOP, w: pw, h: 4.85, radius: 0.1, fill: C.tint, name: 'Example panel' });
    K.text(slide, 'EXAMPLE · API DEMO', { x: px + 0.3, y: TOP + 0.2, w: pw - 0.6, h: 0.28, fontSize: 12, bold: true, color: C.teal, charSpacing: 1.5 });
    K.text(slide, `${req.scenario} · ${req.num_rows} rows · seed ${req.seed}`, { x: px + 0.3, y: TOP + 0.5, w: pw - 0.5, h: 0.36, fontSize: 15, bold: true, fontFace: CODE_FONT, color: C.ink });
    const kv = [
      ['Condition (met exactly from pooled SDV draws)', condText(g.conditions) || '—'],
      ['Sampled → accepted → kept', `${fmtInt(g.sampled_rows)} → ${fmtInt(g.accepted_rows)} (${pct(g.acceptance_rate, 1)} acceptance) → ${fmtInt(g.num_rows_generated)} rows`],
      ['Rejected by rule', rejected.length ? rejected.map(([r, n]) => `${r} ${fmtInt(n)}`).join(', ') : 'none'],
      ['Allocation strata', `${(strata.columns || []).join(' × ')}: ${fmtInt(strata.strata_cells || 0)}${NBSP}cells, ${fmtInt(strata.rebalanced_rows || 0)}${NBSP}rows rebalanced`],
      ['Time', `${f2(g.elapsed_seconds)}${NBSP}s of a ${Math.round(g.time_budget_seconds)}${NBSP}s budget`],
      ['Hard rules', `${opt(v, 'hard_rules.passed', '?')}/${opt(v, 'hard_rules.checked', '?')} passed on all ${fmtInt(v.n_rows)} rows`],
      ['Calendar mode', `${(calReq.wards || []).join(' + ')} × ${calDays} days → ${fmtInt(calBody.n_rows)} rows${calFull ? ', one per ward per day' : ''}`],
    ];
    // label above value, so every value has the full panel width and stays on one line
    kv.forEach(([k, val], i) => {
      const y = TOP + 0.98 + i * 0.54;
      K.text(slide, k, { x: px + 0.3, y, w: pw - 0.6, h: 0.22, fontSize: 12, bold: true, color: C.teal });
      K.text(slide, val, { x: px + 0.3, y: y + 0.22, w: pw - 0.5, h: 0.28, fontSize: 14, color: HEX.dk1 });
    });
  }

  // ---------------------------------------------------------------- 11. Evaluation framework
  {
    const famColors = [HEX.accent1, HEX.dk2, HEX.accent2, HEX.accent6, HEX.accent4, HEX.accent5];
    const desc = {
      fidelity: 'SDV QualityReport: shapes + pair trends',
      validity: 'unfiltered samples passing all hard rules',
      operational: `${opWD.n_checks} ward-day and ${opPS.n_checks} stay checks vs real data`,
      utility: 'ML train-synthetic-test-real + decisions',
      privacy: 'copies and closest-record distances',
    };
    // calibration settings per table (the evaluated size n differs between tables in a full run)
    const noiseOf = (t, k) => opt(ctx.cmp[t].noise, k);
    const perTable = (k, fmt, join) => {
      const [a, b] = TABLES.map((t) => noiseOf(t, k));
      if (a === null && b === null) return null;
      return a === b ? fmt(a) : join(a === null ? '–' : fmt(a), b === null ? '–' : fmt(b));
    };
    const nCal = perTable('reps', String, (a, b) => `${a} / ${b}`);
    const nCalNotes = perTable('reps', String, (a, b) => `${a} (ward_daily) and ${b} (patient_stays)`);
    const nRows = perTable('n', fmtInt, (a, b) => `${a} / ${b}`);
    const nRowsNotes = perTable('n', (x) => `${fmtInt(x)} rows`, () => {
      const [a, b] = TABLES.map((t) => noiseOf(t, 'n'));
      return `${a === null ? '–' : fmtInt(a)} ward-days and ${b === null ? '–' : fmtInt(b)} stays`;
    });
    const cvWD = opt(ctx.cmp.ward_daily.noise, 'real_pass_rate_cv');
    const cvPS = opt(ctx.cmp.patient_stays.noise, 'real_pass_rate_cv');
    const slide = contentSlide({
      member: 'C', kicker: 'Week 3 · evaluation', title: `Scoring: ${word(families.length)} components, two anchors`, seconds: 40,
      source: 'Sources: reports/results/model_comparison.json (composite_weights, references, noise_calibration)',
      notes: [
        `I am Member C. Every model generates through the same engine and is scored on ${word(families.length)} components, weighted into one composite: ${families.map((f) => `${famLabel[f] || f} ${pct(ctx.weights[f])}`).join(', ')}. Raw validity uses unfiltered samples, so it shows what the model itself learned.`,
        `The real holdout, scored like a model, is the ceiling; a no-learning baseline drawing each column independently within its stratum is the floor.`,
        `Operational thresholds are raised to the real-data noise limit from ${nCalNotes} re-splits at the evaluated size of ${nRowsNotes}, so a fresh real sample passes every check in ${cvWD === null ? 'most' : pct(cvWD)} and ${cvPS === null ? 'most' : pct(cvPS)} of draws.`,
      ].join('\n\n'),
    });
    slide.addChart(pres.ChartType.doughnut, [{ name: 'Composite weight', labels: families.map((f) => famLabel[f] || f), values: families.map((f) => ctx.weights[f]) }], Object.assign(chartBase(), {
      x: MX - 0.1, y: TOP + 0.15, w: 3.5, h: 3.5, holeSize: 52, chartColors: famColors.slice(0, families.length),
      // 14 pt bold counts as large text: white keeps >= 3:1 contrast even on the amber and grey slices
      showLegend: false, showPercent: true, showValue: false, showLabel: false, dataLabelColor: 'FFFFFF', dataLabelFontSize: 14, dataLabelFontBold: true,
      showTitle: false,
    }));
    K.text(slide, 'composite\nweights', { x: MX + 0.75, y: TOP + 1.5, w: 1.8, h: 0.8, fontSize: 14, bold: true, align: 'center', valign: 'middle', color: HEX.muted });
    const fx = MX + 3.5;
    const fw = 4.15;
    families.forEach((f, i) => {
      const y = TOP + 0.02 + i * 0.97;
      slide.addShape(pres.ShapeType.ellipse, { x: fx, y: y + 0.08, w: 0.24, h: 0.24, fill: { color: famColors[i] }, line: { type: 'none' } });
      K.text(slide, [{ text: `${famLabel[f] || cap(f)}  `, options: { bold: true } }, { text: pct(ctx.weights[f]), options: { bold: true, color: C.teal } }], { x: fx + 0.36, y, w: fw - 0.36, h: 0.34, fontSize: 16 });
      K.text(slide, desc[f] || '', { x: fx + 0.36, y: y + 0.36, w: fw - 0.36, h: 0.5, fontSize: 14, color: HEX.muted });
    });
    const rx = fx + fw + 0.25;
    const rw = W - MX - rx;
    const cards = [
      ['Ceiling and floor', 'Ceiling: real holdout rows, never used for training, scored like a model. Floor: a no-learning baseline drawing each column independently within its stratum.', 1.75],
      ['Thresholds calibrated to real noise', `${nCal} re-splits of the real data at n = ${nRows} rows: pass if value ≤ max(design threshold, real noise limit). Real samples pass all checks in ${cvWD === null ? '–' : pct(cvWD)} / ${cvPS === null ? '–' : pct(cvPS)} of draws.`, 1.75],
      ['Selection rule', 'Best complete composite per table → registry default used by the API.', 1.05],
    ];
    let cyy = TOP;
    cards.forEach(([head, body, h], i) => {
      K.box(slide, { x: rx, y: cyy, w: rw, h, radius: 0.1, fill: i === 2 ? 'D6EAE7' : C.tint, name: head });
      K.text(slide, head, { x: rx + 0.22, y: cyy + 0.13, w: rw - 0.44, h: 0.32, fontSize: 16, bold: true });
      K.text(slide, body, { x: rx + 0.22, y: cyy + 0.48, w: rw - 0.4, h: h - 0.56, fontSize: 14, color: HEX.muted });
      cyy += h + 0.16;
    });
  }

  // ---------------------------------------------------------------- 12. Results
  {
    const b = TABLES.map((t) => ctx.cmp[t].best);
    const title = b[0] === b[1] ? `${b[0]} scores best on both tables` : `Best: ${b[0]} (ward_daily), ${b[1]} (patient_stays)`;
    const share = (t) => {
      const c = ctx.cmp[t];
      const best = c.models.find((m) => m.id === c.best) || c.models[0];
      const spread = c.ceiling - c.floor;
      const norm = typeof best.normalised === 'number' ? best.normalised : spread >= 0.05 ? (best.composite - c.floor) / spread : null;
      return { best, norm };
    };
    // observations that hold for the data at hand (checked, not assumed)
    const deep = ['ctgan', 'tvae'];
    const deepName = { ctgan: 'CTGAN', tvae: 'TVAE' };
    const deepBelow = TABLES.every((t) => {
      const c = ctx.cmp[t];
      const top = c.models[0];
      return deep.every((id) => { const m = c.models.find((x) => x.id === id); return !m || (m.id !== top.id && m.composite < top.composite); });
    });
    // deep models against the no-learning floor: below it, or level with it at the three decimals the chart shows
    const floorRel = TABLES.map((t) => {
      const c = ctx.cmp[t];
      const ms = deep.map((id) => c.models.find((x) => x.id === id)).filter(Boolean);
      return {
        t,
        n: ms.length,
        below: ms.filter((m) => m.composite < c.floor && f3(m.composite) !== f3(c.floor)).map((m) => m.id),
        level: ms.filter((m) => f3(m.composite) === f3(c.floor)).map((m) => m.id),
      };
    });
    const who = (ids, n) => (ids.length === 2 && n === 2 ? 'both' : listJoin(ids.map((id) => deepName[id])));
    const floorParts = [
      ...floorRel.filter((r) => r.below.length).map((r) => `on ${r.t} ${who(r.below, r.n)} ${r.below.length > 1 ? 'are' : 'is'} even below the floor`),
      ...floorRel.filter((r) => r.level.length).map((r) => `on ${r.t} ${listJoin(r.level.map((id) => deepName[id]))} ${r.level.length > 1 ? 'are' : 'is'} level with the floor`),
    ].map((p, i) => (i > 0 ? p.replace(/ the floor$/, ' it') : p));
    const floorText = floorParts.length > 1 ? `${floorParts.slice(0, -1).join(', ')}, and ${floorParts[floorParts.length - 1]}` : floorParts.join('');
    const narrow = TABLES.filter((t) => ctx.cmp[t].ceiling - ctx.cmp[t].floor < 0.1);
    const deepNote = [
      deepBelow ? `CTGAN and TVAE score below the best GaussianCopula model on both tables${ctx.quick ? ` - in this quick run after only ${ctx.epochs.join('/')} epochs` : ''}${floorText ? `; ${floorText}` : ''}.` : '',
      narrow.length ? `On ${listJoin(narrow)} the floor sits close to the ceiling, so the share of the gap is the fairer reading.` : '',
    ].filter(Boolean).join(' ');
    const slide = contentSlide({
      member: 'C', kicker: 'Week 3 · results', title, seconds: 40,
      source: 'Source: reports/results/model_comparison.json',
      footnote: quick,
      notes: [
        ...TABLES.map((t, i) => {
          const c = ctx.cmp[t];
          const { best, norm } = share(t);
          const next = c.models.find((m) => m.id !== best.id);
          const refs = i === 0 ? `a ceiling of ${f3(c.ceiling)} and a floor of ${f3(c.floor)}` : `${f3(c.ceiling)} and ${f3(c.floor)}`;
          const gap = norm === null ? '' : i === 0 ? ` - ${pct(norm)} of the floor-to-ceiling gap` : ` - ${pct(norm)} of the gap`;
          return `On ${t}, ${best.id} reaches ${f3(best.composite)} against ${refs}${gap}${next ? `; next comes ${next.id} at ${f3(next.composite)}` : ''}.`;
        }),
        deepNote,
      ].filter(Boolean).join('\n\n'),
    });
    const chartW = (CW - 0.3) / 2;
    TABLES.forEach((t, i) => {
      const c = ctx.cmp[t];
      const labels = ['Real holdout (ceiling)', ...c.models.map((m) => m.id), 'No-learning (floor)'];
      const values = [c.ceiling, ...c.models.map((m) => m.composite), c.floor];
      const colors = [HEX.dk2, ...c.models.map((m) => (m.id === c.best ? HEX.accent1 : HEX.accent3)), HEX.accent4];
      slide.addChart(pres.ChartType.bar, [{ name: 'Composite', labels, values }], Object.assign(chartBase(), {
        x: MX + i * (chartW + 0.3), y: TOP, w: chartW, h: 4.25, barDir: 'bar', chartColors: colors,
        catAxisOrientation: 'maxMin', valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.15, valGridLine: { style: 'none' },
        showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.000', barGapWidthPct: 45,
        showLegend: false, showTitle: true, title: `${t} — composite score`,
      }));
      const { best, norm } = share(t);
      K.text(slide, [
        { text: `${best.id} ${f3(best.composite)}`, options: { bold: true, color: C.teal } },
        { text: norm === null ? ` vs ceiling ${f3(c.ceiling)} and floor ${f3(c.floor)}` : `: ${pct(norm)} of the floor-to-ceiling gap (floor ${f3(c.floor)}, ceiling ${f3(c.ceiling)})`, options: {} },
      ], { x: MX + i * (chartW + 0.3) + 0.15, y: TOP + 4.35, w: chartW - 0.3, h: 0.6, fontSize: 14 });
    });
  }

  // ---------------------------------------------------------------- 13. Ablation
  {
    const comp = (t, id) => ctx.cmp[t].models.find((m) => m.id === id);
    for (const t of TABLES) for (const id of ABLATION) if (!comp(t, id)) fail(`model_comparison.json: no scored ${t}/${id} for the ablation slide`);
    const valid = (t, id) => comp(t, id).components.validity;
    const composite = (t, id) => comp(t, id).composite;
    const series = (fn) => TABLES.map((t) => ({ name: t, labels: ABLATION, values: ABLATION.map((id) => fn(t, id)) }));
    const [s0, s1, s2] = ABLATION;
    const segWD = d.cards.ward_daily[s1].n_segments;
    const segPS = d.cards.patient_stays[s2].n_segments;
    const prog2 = (d.cards.ward_daily[s2].constraint_groups || []).filter((g) => ctx.groupKind[g] === 'programmable').length;
    const allOutputValid = TABLES.every((t) => d.mc.tables[t].models.filter((m) => m.status === 'ok').every((m) => m.output_validity === 1));
    // ward_daily marginals grouped by a band recomputed from the counts (model card of the tuned model)
    const bandNames = ctx.defaults.ward_daily === s2 ? ctx.bandGroups.map((b) => human(b).replace(/ level$/, '')) : [];
    // A/B test of the tuned ward_daily spec across evaluation seeds (only when it describes the default)
    const ab = ctx.ab && ctx.defaults.ward_daily === s2 ? ctx.ab : null;
    const span = (s, fmt) => (fmt(s.min) === fmt(s.max) ? fmt(s.min) : `${fmt(s.min)} to ${fmt(s.max)}`);
    const spanCount = (s) => (s.min === s.max ? String(s.min) : `${s.min} ${s.max - s.min === 1 ? 'or' : 'to'} ${s.max}`);
    const spanPct = (s) => (pct(s.min, 1) === pct(s.max, 1) ? pct(s.min, 1) : `${pct(s.min, 1).replace('%', '')} to ${pct(s.max, 1)}`);
    const abNote = ab
      ? `An A/B test on ${word(ab.n)} evaluation seeds compared ${bandNames.length ? 'those band-grouped marginals' : `this ward_daily ${s2} spec`} with the previous spec: composite ${span(ab.adopt.composite, f3)}, mean ${f3(ab.adopt.composite.mean)}, against ${span(ab.prev.composite, f3)}, mean ${f3(ab.prev.composite.mean)}; ${spanCount(ab.adopt.passed)} of ${ab.scored} checks pass instead of ${spanCount(ab.prev.passed)}; raw validity is ${spanPct(ab.adopt.valid)} instead of ${spanPct(ab.prev.valid)}${allOutputValid && ab.adopt.valid.mean < ab.prev.valid.mean ? ', a cost rejection absorbs' : ''}.`
      : '';
    const slide = contentSlide({
      member: 'C', kicker: 'Week 3 · improvements', title: 'What improved SDV generation', seconds: 50,
      source: 'Sources: reports/results/model_comparison.json, models/registry.json',
      footnote: quick,
      notes: [
        `The ablation adds one block per step, evaluated identically: left raw validity, right the composite.`,
        `${s0}, one global copula without constraints, passes every hard rule in only ${pct(valid('ward_daily', s0), 1)} of ward-days and ${pct(valid('patient_stays', s0), 1)} of stays, composite ${f3(composite('ward_daily', s0))} and ${f3(composite('patient_stays', s0))}. ${s1} adds segments, built-in constraints, ordered encodings and tuned marginals: ${pct(valid('ward_daily', s1))} and ${pct(valid('patient_stays', s1))} valid, composite ${f3(composite('ward_daily', s1))} and ${f3(composite('patient_stays', s1))}. ${s2} adds the ${word(prog2)} programmable groups${bandNames.length ? `, including ward_daily marginals per recomputed ${listJoin(bandNames)} band` : ''}: ${pct(valid('ward_daily', s2))} and ${pct(valid('patient_stays', s2))}${ab ? '' : `, composite ${f3(composite('ward_daily', s2))} and ${f3(composite('patient_stays', s2))}`}.`,
        // without the A/B evidence: why raw validity matters although rejection delivers only valid rows
        abNote || `${allOutputValid ? 'Rejection makes all delivered output valid, but a' : 'A'} valid model wastes fewer samples and keeps its learned distribution.`,
      ].join('\n\n'),
    });
    const chartW = (CW - 0.3) / 2;
    const common = () => Object.assign(chartBase(), {
      y: TOP, w: chartW, h: 3.25, barDir: 'col', barGrouping: 'clustered', chartColors: [HEX.accent1, HEX.accent2],
      valAxisHidden: true, valAxisMinVal: 0, valGridLine: { style: 'none' }, showValue: true, dataLabelPosition: 'outEnd',
      showLegend: true, legendPos: 't', showTitle: true, barGapWidthPct: 60, catAxisLabelFontFace: CODE_FONT, catAxisLabelFontSize: 13,
    });
    slide.addChart(pres.ChartType.bar, series(valid), Object.assign(common(), { x: MX, valAxisMaxVal: 1.18, dataLabelFormatCode: '0.0%', title: 'Raw hard-valid rate (unfiltered samples)' }));
    slide.addChart(pres.ChartType.bar, series(composite), Object.assign(common(), { x: MX + chartW + 0.3, valAxisMaxVal: 1.18, dataLabelFormatCode: '0.000', title: 'Composite score' }));
    const cards = [
      [s0, 'One global GaussianCopula, default encodings, no constraints (Week 2 baseline).'],
      [s1, `+ per-ward segments (${segWD}), SDV built-in constraints, ordinal metadata, tuned marginals.`],
      [s2, `+ ${word(prog2)} programmable constraint groups; ward × source segments (${segPS}) for stays.`],
    ];
    const cw = (CW - 2 * 0.25) / 3;
    cards.forEach(([id, txt], i) => {
      const x = MX + i * (cw + 0.25);
      const y = TOP + 3.45;
      K.box(slide, { x, y, w: cw, h: 1.45, radius: 0.1, fill: i === 2 ? 'D6EAE7' : C.tint, name: `Step ${id}` });
      K.text(slide, [{ text: `${i + 1}. `, options: { bold: true, color: C.teal } }, { text: id, options: { bold: true, fontFace: CODE_FONT, color: C.ink } }], { x: x + 0.2, y: y + 0.14, w: cw - 0.4, h: 0.34, fontSize: 15 });
      K.text(slide, txt, { x: x + 0.2, y: y + 0.52, w: cw - 0.4, h: 0.86, fontSize: 14, color: HEX.muted });
      if (i < cards.length - 1) K.line(slide, x + cw + 0.03, y + 0.72, x + cw + 0.22, y + 0.72, { color: HEX.accent1, width: 2, arrow: true });
    });
  }

  // ---------------------------------------------------------------- 14. Occupancy and scheduling patterns
  {
    const check = (op, name) => {
      const c = op.checks.find((x) => x.name === name);
      if (!c) fail(`operational results have no check "${name}"`);
      return c;
    };
    const occ = check(opWD, 'occupancy_mean_by_ward');
    const elec = check(opWD, 'weekly_elective_profile');
    const dis = check(opPS, 'discharge_hour_profile');
    const wardIds = ctx.wards.map((w) => w.id).filter((w) => w in occ.real);
    const days = ctx.weekdays.filter((x) => x in elec.real);
    const hours = Object.keys(dis.real).map(Number).sort((a, b) => a - b);
    const peak = (profile) => Object.entries(profile).sort((a, b) => b[1] - a[1])[0][0];
    const dayName = (code) => ({ Mon: 'Mondays', Tue: 'Tuesdays', Wed: 'Wednesdays', Thu: 'Thursdays', Fri: 'Fridays', Sat: 'Saturdays', Sun: 'Sundays' }[code] || code);
    const failed = (op) => (op.failed_checks || []).map(checkLabel);
    const passText = (op) => `${op.n_passed}/${op.n_scored} scored checks pass`;
    // plain-language reading of failed checks, direction taken from the real vs synthetic values
    const plainGaps = [];
    for (const op of [opWD, opPS]) {
      for (const name of op.failed_checks || []) {
        const c = op.checks.find((x) => x.name === name) || {};
        const r = c.real;
        const s = c.synthetic;
        if (/^relationship_/.test(name) && typeof r === 'number' && typeof s === 'number') {
          const what = String(c.description || human(name)).replace(/^Spearman correlation of /i, '').replace(/\s*\(.*$/, '').replace(/\.$/, '');
          plainGaps.push(`the correlation of ${what} is ${Math.abs(s) < Math.abs(r) ? 'weaker' : 'stronger'} than in real data (${f2(s)} vs ${f2(r)})`);
        } else if (name === 'weekend_weekday_discharge_ratio' && typeof r === 'number' && typeof s === 'number') {
          plainGaps.push(`weekend discharges are ${s > r ? 'too frequent' : 'too rare'} relative to weekdays (ratio ${f2(s)} vs ${f2(r)})`);
        } else if (r && s && typeof r === 'object' && typeof s === 'object') {
          const keys = Object.keys(r).filter((k) => typeof r[k] === 'number' && typeof s[k] === 'number');
          if (keys.length > 1) {
            const spread = (o) => Math.max(...keys.map((k) => o[k])) - Math.min(...keys.map((k) => o[k]));
            plainGaps.push(`${human(name)} varies ${spread(s) < spread(r) ? 'less' : 'more'} across groups than in real data`);
          }
        }
      }
    }
    const sat = days[days.length - 2];
    // failing checks first; otherwise checks that pass their threshold but lie beyond real-data noise
    const anyFailed = failed(opWD).length + failed(opPS).length > 0;
    const beyond = TABLES.map((t) => [t, ctx.opGaps[t].beyondNoise.map(checkLabel)]).filter(([, l]) => l.length);
    let opTail;
    if (anyFailed) {
      opTail = `. Failing: ${[failed(opWD).length ? `on ward_daily ${listJoin(failed(opWD))}` : '', failed(opPS).length ? `on patient_stays ${listJoin(failed(opPS))}` : ''].filter(Boolean).join('; ')}.${plainGaps.length ? ` In plain words: ${plainGaps.join('; ')}.` : ''}`;
    } else if (beyond.length) {
      opTail = `; but ${listJoin(beyond.map(([t, l]) => `${word(l.length)} ${t} check${l.length === 1 ? '' : 's'} - ${listJoin(l)} -`))} differ from real data by more than real samples do.`;
    } else {
      const allWithin = [opWD, opPS].every((op) => typeof op.n_within_real_noise === 'number' && op.n_within_real_noise >= op.n_scored);
      opTail = allWithin ? ', and every value lies within real-data noise.' : '; no check fails.';
    }
    // box line per table: failing checks, else passing checks beyond real-data noise, else none
    const checkLine = (op, t) => {
      const f = failed(op);
      const b = ctx.opGaps[t].beyondNoise.map(checkLabel);
      if (f.length) return { text: `Failing: ${f.join('; ')}`, color: C.red, fill: HEX.tintRed };
      if (b.length) return { text: `Beyond real-data noise: ${b.join(', ')}`, color: HEX.muted, fill: HEX.tintAmber };
      const within = typeof op.n_within_real_noise === 'number' && op.n_within_real_noise >= op.n_scored;
      return { text: within ? 'No failing check; every value within real-data noise' : 'No failing check', color: HEX.muted, fill: C.tint };
    };
    const slide = contentSlide({
      member: 'C', kicker: 'Week 3 · operational patterns', title: 'Occupancy and scheduling vs real data', seconds: 35,
      source: `Source: operational checks in reports/results/ward_daily__${dWD}.json and patient_stays__${dPS}.json`,
      footnote: quick,
      notes: [
        `High fidelity can still hide wrong bed-management patterns, so we compare them directly. Mean occupancy per ward differs by at most ${(100 * Math.max(...wardIds.map((w) => Math.abs(occ.real[w] - occ.synthetic[w])))).toFixed(1)} percentage points, ${dayName(sat).replace(/s$/, '')} elective lists average ${f2(elec.real[sat])} real versus ${f2(elec.synthetic[sat])} synthetic, and discharges peak at ${peak(dis.real) === peak(dis.synthetic) ? `${peak(dis.real)}:00 in both` : `${peak(dis.real)}:00 real and ${peak(dis.synthetic)}:00 synthetic`}.`,
        `With noise-calibrated thresholds, ${opWD.n_passed} of ${opWD.n_scored} checks pass on ward_daily and ${opPS.n_passed} of ${opPS.n_scored} on patient_stays${opTail}`,
      ].join('\n\n'),
    });
    const cw = (CW - 2 * 0.28) / 3;
    const ch = 3.45;
    const base = () => Object.assign(chartBase(), { y: TOP, w: cw, h: ch, showLegend: true, legendPos: 'b', showTitle: true, chartColors: [HEX.accent4, HEX.accent1] });
    slide.addChart(pres.ChartType.bar, [
      { name: 'Real', labels: wardIds, values: wardIds.map((w) => occ.real[w]) },
      { name: 'Synthetic', labels: wardIds, values: wardIds.map((w) => occ.synthetic[w]) },
    ], Object.assign(base(), { x: MX, barDir: 'col', barGapWidthPct: 50, title: 'Mean midnight occupancy by ward', valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: '0%', valAxisMajorUnit: 0.25, catAxisLabelFontSize: 11 }));
    slide.addChart(pres.ChartType.bar, [
      { name: 'Real', labels: days, values: days.map((x) => elec.real[x]) },
      { name: 'Synthetic', labels: days, values: days.map((x) => elec.synthetic[x]) },
    ], Object.assign(base(), { x: MX + cw + 0.28, barDir: 'col', barGapWidthPct: 50, title: 'Electives per ward-day by weekday', valAxisMinVal: 0, valAxisLabelFormatCode: '0.0' }));
    slide.addChart(pres.ChartType.line, [
      { name: 'Real', labels: hours.map(String), values: hours.map((h) => dis.real[h]) },
      { name: 'Synthetic', labels: hours.map(String), values: hours.map((h) => dis.synthetic[h]) },
    ], Object.assign(base(), { x: MX + 2 * (cw + 0.28), title: 'Discharges by hour of day (stays)', lineSize: 2.25, lineDataSymbol: 'none', valAxisMinVal: 0, valAxisLabelFormatCode: '0%', catAxisLabelFrequency: 3, catAxisLabelFontSize: 11 }));
    const by = TOP + ch + 0.22;
    const bh = 1.2;
    const lw = 7.6;
    const lineWD = checkLine(opWD, 'ward_daily');
    const linePS = checkLine(opPS, 'patient_stays');
    K.box(slide, { x: MX, y: by, w: lw, h: bh, radius: 0.1, fill: lineWD.fill, name: 'ward_daily checks' });
    K.text(slide, [
      { text: `ward_daily · ${passText(opWD)}`, options: { bold: true, breakLine: true } },
      { text: lineWD.text, options: { color: lineWD.color } },
    ], { x: MX + 0.22, y: by + 0.1, w: lw - 0.44, h: bh - 0.2, fontSize: 14, valign: 'middle' });
    K.box(slide, { x: MX + lw + 0.25, y: by, w: CW - lw - 0.25, h: bh, radius: 0.1, fill: linePS.fill, name: 'patient_stays checks' });
    K.text(slide, [
      { text: `patient_stays · ${passText(opPS)}`, options: { bold: true, breakLine: true } },
      { text: linePS.text, options: { color: linePS.color } },
    ], { x: MX + lw + 0.47, y: by + 0.1, w: CW - lw - 0.69, h: bh - 0.2, fontSize: 14, valign: 'middle' });
  }

  // ---------------------------------------------------------------- 15. Scenario coverage
  {
    const covWD = ctx.coverage.ward_daily;
    const covPS = ctx.coverage.patient_stays;
    const nEval = covWD.n_scenarios + covPS.n_scenarios;
    const nOk = covWD.n_succeeded + covPS.n_succeeded;
    const demoCov = opt(d.demo, 'scenario_coverage', []);
    const demoOk = demoCov.filter((s) => s.outcome === 'ok' && s.all_hard_rules_passed).length;
    const allScen = TABLES.flatMap((t) => ctx.coverage[t].scenarios);
    const inScen = Math.min(...allScen.map((s) => (typeof s.in_scenario_rate === 'number' ? s.in_scenario_rate : 0)));
    const rowsPer = opt(covWD, 'rows_per_scenario');
    const weakest = ctx.weakestScenarios;
    const weakFiltered = ctx.weakFiltered;
    const allOk = nOk === nEval && demoOk === demoCov.length && demoCov.length > 0;
    const slide = contentSlide({
      member: 'C', kicker: 'Week 3 · scenario coverage', seconds: 35,
      title: allOk ? 'Every scenario can be generated' : `${nOk} of ${nEval} scenarios generated`,
      source: `Sources: scenario_coverage in reports/results (${dWD}), reports/api_demo/summary.json`,
      footnote: quick,
      notes: [
        `${allOk ? 'Every scenario can be generated' : 'Scenario coverage'}: ${nOk} of ${nEval} in the evaluation at ${rowsPer} rows each, ${inScen === 1 ? 'every delivered row' : `at least ${pct(inScen)} of rows`} inside its scenario, and ${demoOk} of ${demoCov.length} again through the API demo. Each bar averages coverage - fulfilment and the ward and band mix - with fidelity to the real scenario subset; the table scores are ${f2(covWD.score)} and ${f2(covPS.score)}.`,
        `The weakest, ${listJoin(weakest.map((s) => `${s.name} (${f2(s.score)})`))}, ${weakFiltered ? 'are filtered scenarios: few raw rows pass, and some land on other ward and band combinations than in the real subset' : 'are listed in the Week 3 report'}.`,
      ].join('\n\n'),
    });
    const lw = 3.45;
    const stats = [
      [`${nOk}/${nEval}`, `scenarios generated in the evaluation (${rowsPer} rows each)`],
      [pct(inScen), 'of delivered rows inside their scenario'],
      [`${f2(covWD.score)} / ${f2(covPS.score)}`, 'coverage score, ward_daily / patient_stays'],
      [`${demoOk}/${demoCov.length}`, 'scenarios via the API demo, all hard rules passed'],
    ];
    stats.forEach(([v, l], i) => K.stat(slide, MX, TOP - 0.05 + i * 1.22, lw, v, l, { size: 32, valueH: 0.56, labelH: 0.56 }));
    const cx = MX + lw + 0.3;
    const cwTot = W - MX - cx;
    const chartW = [cwTot * 0.53, cwTot * 0.47];
    let x = cx;
    TABLES.forEach((t, i) => {
      const scen = ctx.coverage[t].scenarios.filter((s) => typeof s.score === 'number');
      const titleOf = (s) => cap(human(s.name));
      slide.addChart(pres.ChartType.bar, [{ name: 'Scenario score', labels: scen.map(titleOf), values: scen.map((s) => s.score) }], Object.assign(chartBase(), {
        x, y: TOP, w: chartW[i] - 0.1, h: 4.88, barDir: 'bar', chartColors: [HEX.accent1], catAxisOrientation: 'maxMin',
        valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.2, valGridLine: { style: 'none' }, showValue: true,
        dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelFontSize: 11, catAxisLabelFontSize: 11,
        showLegend: false, showTitle: true, title: `${t} (${scen.length})`, barGapWidthPct: 35,
      }));
      x += chartW[i];
    });
  }

  // ---------------------------------------------------------------- 16. Privacy and utility
  {
    const priv = (t) => pick(d.results[t][ctx.defaults[t]], 'privacy', `${t} privacy results`);
    const pWD = priv('ward_daily');
    const pPS = priv('patient_stays');
    const exactRows = (p) => Math.round(num(pick(p, 'exact_matches.synthetic_rate', 'privacy'), 'exact match rate') * num(pick(p, 'n_rows.synthetic_used', 'privacy'), 'synthetic rows'));
    const fam = ctx.famScore;
    const optWD = pick(resWD, 'optimization', 'ward_daily optimisation');
    const optFloor = opt(d.refs.ward_daily.floor.result, 'optimization', {});
    const avail = ctx.utilityFamilies;
    // families below the floor; "level" = below it only beyond the third decimal
    const gaps = ctx.utilityGaps.map((g) => (g.level ? `${g.label}: level with the floor (${f3(g.m)})` : `${g.label}: ${f3(g.m)} < floor ${f3(g.f)}`));
    const gapNotes = ctx.utilityGaps.map((g) => `${g.k === 'ml_utility' ? 'machine-learning utility' : 'optimisation utility'} on ${g.t} is ${g.level ? `only level with the no-learning floor (${f3(g.m)})` : `${f3(g.m)} against ${f3(g.f)} for the no-learning floor`}`);
    const polM = opt(optWD, 'policy_evaluation.objective_accuracy');
    const polF = opt(optFloor, 'policy_evaluation.objective_accuracy');
    const regret = opt(optWD, 'booking_quota.relative_regret');
    const regretFloor = opt(optWD, 'booking_quota.real_decision_regret_vs_oracle');
    const agg = TABLES.map((t) => opt(priv(t), 'aggregate_disclosure.columns', [])).map((cols) => cols.map((c) => human(c).replace(' id', '')).join(' × '));
    const aggMirror = TABLES.every((t) => opt(priv(t), 'aggregate_disclosure.mirrors_training_frequencies') === true);
    const slide = contentSlide({
      member: 'C', kicker: 'Week 3 · privacy and utility', title: 'Privacy and utility: strengths and gaps', seconds: 40,
      source: `Sources: privacy, ml_utility and optimization in reports/results (${dWD} vs references)`,
      footnote: quick,
      notes: [
        `Privacy: ${fmtInt(pWD.n_rows.synthetic_used)} synthetic ward-days and ${fmtInt(pPS.n_rows.synthetic_used)} stays contain ${exactRows(pWD) + exactRows(pPS) === 0 ? 'no' : `${exactRows(pWD)} and ${exactRows(pPS)}`} exact training copies, and their median distance to the closest training record is ${f2(pWD.dcr.median_ratio)} and ${f2(pPS.dcr.median_ratio)} times that of unseen real rows${pWD.dcr.median_ratio >= 1 && pPS.dcr.median_ratio >= 1 ? ' - no closer than new real data' : ''}. The caveat: the allocation strata ${aggMirror ? 'copy' : 'follow'} the training mix of wards and bands${aggMirror ? ' exactly' : ''}, an aggregate disclosure we report.`,
        `Utility: machine-learning utility is ${f2(fam('ward_daily', 'model', 'ml_utility'))} and ${f2(fam('patient_stays', 'model', 'ml_utility'))}; ${gapNotes.length ? `but ${gapNotes.join('; ')}` : 'no family is below the floor'}.${polM !== null && polF !== null ? ` Policy-evaluation accuracy is ${f2(polM)} against ${f2(polF)} for the floor` : ''}${regret !== null && regretFloor !== null ? `${polM !== null && polF !== null ? ', and' : ''} synthetic booking quotas lose ${pct(regret, 1)} on the real holdout against ${pct(regretFloor, 1)} for real-data quotas` : ''}${(polM !== null && polF !== null) || (regret !== null && regretFloor !== null) ? '.' : ''}${polM !== null && polF !== null && polM < polF ? ' The link between elective lists and bed pressure is weaker than in the real data.' : ''}`,
      ].join('\n\n'),
    });
    const lw = 5.85;
    K.icon(slide, 'lock', MX, TOP, 0.56);
    K.text(slide, 'Privacy', { x: MX + 0.72, y: TOP + 0.08, w: 3, h: 0.4, fontSize: 19, bold: true });
    // quoted from the results; "(expected 0.05)" is shown as a percentage like the values next to it
    const metricText = (key, fallback, asPct = false) => {
      const text = String(opt(pWD, `metric_descriptions.${key}`, fallback)).replace(/>=/g, '≥').replace(/<=/g, '≤');
      return asPct ? text.replace(/\(expected (\d*\.\d+)\)/, (m, v) => `(expected ${pct(Number(v))})`) : text;
    };
    const rows = [
      [`${exactRows(pWD)} / ${exactRows(pPS)}`, `exact copies of training rows in ${fmtInt(pWD.n_rows.synthetic_used)} / ${fmtInt(pPS.n_rows.synthetic_used)} synthetic rows`],
      [`${f2(pWD.dcr.median_ratio)} / ${f2(pPS.dcr.median_ratio)}`, metricText('dcr_median_ratio', 'median distance to the closest training record, synthetic vs unseen real rows')],
      [`${pct(pWD.dcr.close_record_share, 1)} / ${pct(pPS.dcr.close_record_share, 1)}`, metricText('close_record_share', 'share of synthetic rows unusually close to a training row', true)],
    ];
    rows.forEach(([v, l], i) => {
      const y = TOP + 0.7 + i * 0.94;
      K.text(slide, v, { x: MX, y, w: 2.3, h: 0.84, fontSize: 24, bold: true, color: C.teal, fontFace: HEAD_FONT, valign: 'middle' });
      K.text(slide, l, { x: MX + 2.4, y, w: lw - 2.4, h: 0.84, fontSize: 14, color: HEX.muted, valign: 'middle' });
    });
    K.box(slide, { x: MX, y: TOP + 3.58, w: lw, h: 1.32, radius: 0.1, fill: HEX.tintAmber, name: 'Privacy gap' });
    K.text(slide, [
      { text: 'Gap: ', options: { bold: true, color: 'A35A12' } },
      { text: `allocation strata ${aggMirror ? 'copy' : 'follow'} the training mix of ${listJoin(agg)}${aggMirror ? ' exactly' : ''} (aggregate disclosure, reported, not scored). Reference data is simulated.`, options: {} },
    ], { x: MX + 0.2, y: TOP + 3.64, w: lw - 0.4, h: 1.2, fontSize: 14, valign: 'middle' });
    const rx = MX + lw + 0.4;
    const rw = W - MX - rx;
    K.icon(slide, 'chart', rx, TOP, 0.56);
    K.text(slide, 'Utility vs references', { x: rx + 0.72, y: TOP + 0.08, w: 4, h: 0.4, fontSize: 19, bold: true });
    const lbl = avail.map(([l]) => l);
    slide.addChart(pres.ChartType.bar, [
      { name: `${sameDefault ? dWD : 'default model'}`, labels: lbl, values: avail.map(([, t, k]) => fam(t, 'model', k)) },
      { name: 'No-learning floor', labels: lbl, values: avail.map(([, t, k]) => fam(t, 'floor', k)) },
      { name: 'Real holdout (ceiling)', labels: lbl, values: avail.map(([, t, k]) => fam(t, 'ceiling', k)) },
    ], Object.assign(chartBase(), {
      x: rx, y: TOP + 0.6, w: rw, h: 2.95, barDir: 'bar', barGrouping: 'clustered', catAxisOrientation: 'maxMin',
      chartColors: [HEX.accent1, HEX.accent4, HEX.dk2], valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.18,
      valGridLine: { style: 'none' }, showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelFontSize: 11,
      showLegend: true, legendPos: 'b', showTitle: false, barGapWidthPct: 40,
    }));
    const gapLines = [...gaps];
    const polGap = polM !== null && polF !== null && polM < polF;
    const regretGap = regret !== null && regretFloor !== null && regret > regretFloor;
    if (polGap) gapLines.push(`Policy-evaluation accuracy: ${f2(polM)} vs floor ${f2(polF)}`);
    if (regretGap) gapLines.push(`Booking-quota regret: ${pct(regret, 1)} vs ${pct(regretFloor, 1)} with real data`);
    if (gapLines.length) {
      K.box(slide, { x: rx, y: TOP + 3.58, w: rw, h: 1.32, radius: 0.1, fill: HEX.tintAmber, name: 'Utility gaps' });
      // one gap per line (four fit); any further gaps share the last line
      const shown = gapLines.length > 4 ? [...gapLines.slice(0, 3), gapLines.slice(3).join('; ')] : gapLines;
      K.text(slide, [
        { text: 'Gaps: ', options: { bold: true, color: 'A35A12' } },
        ...shown.map((t, i) => ({ text: t, options: { breakLine: i < shown.length - 1 } })),
      ], { x: rx + 0.2, y: TOP + 3.64, w: rw - 0.4, h: 1.2, fontSize: 14, valign: 'middle' });
    } else {
      // no utility family below the floor: the decision results that are not gaps are shown as strengths,
      // next to the utility family furthest below the real-data ceiling (above the floor is not "no gap")
      const strengths = [];
      if (polM !== null && polF !== null) strengths.push(`Policy-evaluation accuracy ${f2(polM)} vs floor ${f2(polF)}`);
      if (regret !== null && regretFloor !== null) strengths.push(`Booking-quota regret ${pct(regret, 1)} vs ${pct(regretFloor, 1)} with real data`);
      const belowCeiling = avail.map(([l, t, k]) => ({ l, m: fam(t, 'model', k), c: fam(t, 'ceiling', k) }))
        .filter((g) => typeof g.m === 'number' && typeof g.c === 'number' && g.c > g.m)
        .sort((a, b) => (b.c - b.m) - (a.c - a.m))[0];
      K.box(slide, { x: rx, y: TOP + 3.58, w: rw, h: 1.32, radius: 0.1, fill: C.tint, name: 'Utility strengths' });
      K.text(slide, [
        { text: 'Above the floor: ', options: { bold: true, color: C.teal } },
        { text: `all ${word(avail.length)} utility scores${belowCeiling ? `; furthest from the ceiling: ${belowCeiling.l} ${f2(belowCeiling.m)} vs ${f2(belowCeiling.c)}` : ''}`, options: { breakLine: strengths.length > 0 } },
        ...strengths.map((t, i) => ({ text: t, options: { breakLine: i < strengths.length - 1, color: HEX.muted } })),
      ], { x: rx + 0.2, y: TOP + 3.64, w: rw - 0.4, h: 1.2, fontSize: 14, valign: 'middle' });
    }
  }

  // ---------------------------------------------------------------- 17. FastAPI
  {
    const req = pick(d.demoGenerate.evidence, 'request', 'generation evidence');
    const body = pick(d.demoGenerate.evidence, 'response.body', 'generation evidence');
    const g = body.generation;
    const v = body.validation;
    const q = body.quality || {};
    const groupOf = (p) => {
      if (p === '/' || p === '/health' || p === '/ui') return 'service';
      if (/\/(tables|scenarios|models)/.test(p)) return 'catalogue';
      if (/\/generate/.test(p)) return 'generation';
      if (/\/(validate|evaluate)/.test(p)) return 'quality';
      if (/\/datasets/.test(p)) return 'datasets';
      return 'other';
    };
    const groups = {};
    ctx.endpoints.forEach((e) => { groups[groupOf(e.path)] = (groups[groupOf(e.path)] || 0) + 1; });
    const slide = contentSlide({
      member: 'D', kicker: 'Week 4 · FastAPI generator', title: 'One API to generate, validate, evaluate', seconds: 55,
      source: `Request and response: API demo step ${d.demoGenerate.step.index} (reports/api_demo/${d.demoGenerate.step.evidence}); endpoints from GET /`,
      notes: [
        `I am Member D. The FastAPI service exposes ${ctx.endpoints.length} endpoints in ${word(Object.keys(groups).length)} groups: health and an offline demo UI; the catalogue of tables, rules, scenarios and models; generation; validate and evaluate; and stored datasets to list, download as CSV or JSON, and delete.`,
        `On the left is a real demo request: ${req.json.num_rows} ${req.json.scenario} ward-days with a fixed seed and evaluation on. The response carries the records plus metadata: dataset id and links, the model, the generation report - acceptance ${pct(g.acceptance_rate, 1)}, rejections per rule, strata - the validation summary with ${v.hard_rules.passed} of ${v.hard_rules.checked} hard rules passed, and a quality score of ${f3(q.score)} against ${fmtInt(opt(q, 'reference.reference_rows', 0))} matching real rows.`,
        `Requests run under a time budget, every error has the same JSON shape, and CSV output carries the key metadata in X-BedSynth headers.`,
      ].join('\n\n'),
    });
    const lw = 6.3;
    K.text(slide, `${req.method} ${req.path}`, { x: MX, y: TOP - 0.02, w: lw, h: 0.34, fontSize: 15, bold: true, fontFace: CODE_FONT, color: C.teal });
    // compact JSON: key/value pairs joined greedily into lines of at most ~46 characters
    const pairs = Object.entries(req.json).map(([k, val]) => `${JSON.stringify(k)}: ${JSON.stringify(val)}`);
    const reqLines = [];
    pairs.forEach((p, i) => {
      const piece = p + (i < pairs.length - 1 ? ',' : '');
      const last = reqLines.length - 1;
      if (last >= 0 && (reqLines[last] + ' ' + piece).length <= 46) reqLines[last] += ` ${piece}`;
      else reqLines.push(piece);
    });
    reqLines[0] = `{${reqLines[0]}`;
    for (let i = 1; i < reqLines.length; i += 1) reqLines[i] = ` ${reqLines[i]}`;
    reqLines[reqLines.length - 1] += '}';
    const boxH = 0.25 * reqLines.length + 0.26;
    K.box(slide, { x: MX, y: TOP + 0.38, w: lw, h: boxH, radius: 0.08, fill: C.ink, name: 'Request' });
    K.text(slide, reqLines.join('\n'), { x: MX + 0.25, y: TOP + 0.38 + 0.13, w: lw - 0.4, h: boxH - 0.2, fontSize: 14, fontFace: CODE_FONT, color: C.bg });
    const ry = TOP + 0.38 + boxH + 0.3;
    K.text(slide, 'Response metadata (returned with the records)', { x: MX, y: ry, w: lw, h: 0.32, fontSize: 15, bold: true });
    const rejected = Object.entries(g.rejected_by_rule || {}).map(([r, n]) => `${r} ${n}`).join(', ');
    const kv = [
      ['dataset_id', `${body.dataset_id} · self and CSV / JSON download links`],
      ['model', `${body.model.id} · ${String(body.model.synthesizer).replace(/Synthesizer$/, '')} · SDV ${body.model.sdv_version}`],
      ['generation', `${g.num_rows_generated}/${g.num_rows_requested} rows · acceptance ${pct(g.acceptance_rate, 1)}${rejected ? ` · rejected ${rejected}` : ''}`],
      ['validation', `${v.hard_rules.passed}/${v.hard_rules.checked} hard rules passed · ${v.soft_rules.passed}/${v.soft_rules.checked} soft rules clean`],
      ['quality', `quick score ${f3(q.score)} against ${fmtInt(opt(q, 'reference.reference_rows', 0))} matching real rows`],
    ];
    kv.forEach(([k, val], i) => {
      const y = ry + 0.4 + i * 0.5;
      K.text(slide, k, { x: MX, y, w: 1.45, h: 0.48, fontSize: 14, fontFace: CODE_FONT, bold: true, color: C.ink, valign: 'middle' });
      K.text(slide, val, { x: MX + 1.5, y, w: lw - 1.5, h: 0.48, fontSize: 14, valign: 'middle' });
    });
    const rx = MX + lw + 0.3;
    const rw = W - MX - rx;
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.bg, fill: { color: C.ink } } });
    const rows = [[hdr('Method'), hdr(`Endpoint (${ctx.endpoints.length})`)]].concat(ctx.endpoints.map((e, i) => [
      { text: e.method, options: { fontFace: CODE_FONT, bold: true, fill: { color: i % 2 ? 'FFFFFF' : HEX.lt2 } } },
      { text: e.path, options: { fontFace: CODE_FONT, fill: { color: i % 2 ? 'FFFFFF' : HEX.lt2 } } },
    ]));
    slide.addTable(rows, { x: rx, y: TOP - 0.02, w: rw, colW: [1.0, rw - 1.0], fontSize: 14, rowH: 0.3, border: { type: 'solid', pt: 0.5, color: HEX.rule }, valign: 'middle', margin: [0.0, 0.06, 0.0, 0.08] });
  }

  // ---------------------------------------------------------------- 18. Testing evidence
  {
    const j = d.junit;
    const modules = {};
    j.cases.forEach((c) => {
      const mod = (c.classname.split('.').pop() || 'other').replace(/^test_/, '');
      modules[mod] = (modules[mod] || 0) + 1;
    });
    const mods = Object.entries(modules).sort((a, b) => b[1] - a[1]);
    const steps = ctx.demoSteps;
    const total = num(pick(d.demo, 'total', 'api_demo/summary.json'), 'demo total');
    const passed = num(pick(d.demo, 'passed', 'api_demo/summary.json'), 'demo passed');
    const errorSteps = steps.filter((s) => s.expected_status >= 400);
    const errorCodes = [...new Set(errorSteps.map((s) => s.expected_status))].sort();
    const scenarioSteps = opt(d.demo, 'scenario_coverage', []);
    const reproduced = steps.some((s) => (s.checks || []).some((c) => /same seed/i.test(c.check || '') && c.passed));
    // listed endpoints (GET /) that at least one demo step called, path parameters matched by pattern
    const toRegex = (p) => new RegExp(`^${p.replace(/[.*+?^$()|[\]\\]/g, '\\$&').replace(/\{[^}/]+\}/g, '[^/]+')}$`);
    const callsTo = (e) => steps.some((s) => s.method === e.method && toRegex(e.path).test(String(s.path).split('?')[0]));
    const covered = ctx.endpoints.filter(callsTo);
    const extraPaths = [...new Set(steps.filter((s) => !ctx.endpoints.some((e) => s.method === e.method && toRegex(e.path).test(String(s.path).split('?')[0]))).map((s) => s.path))];
    const scenOk = scenarioSteps.filter((s) => s.outcome === 'ok' && s.all_hard_rules_passed).length;
    const jPassed = j.tests - j.failures - j.errors - j.skipped;
    const broken = j.cases.filter((c) => c.status === 'failed' || c.status === 'error').map((c) => `${c.classname.split('.').pop()}::${c.name}`);
    const slow = d.junitSlow;
    const slowPassed = slow ? slow.tests - slow.failures - slow.errors - slow.skipped : null;
    const slide = contentSlide({
      member: 'D', kicker: 'Week 4 · testing', seconds: 65, timeExtra: 'then about 3 minutes of live demo',
      title: `Testing: ${fmtInt(j.tests)}${slow ? ` + ${fmtInt(slow.tests)}` : ''} tests, ${total} API demo steps`,
      source: `Sources: reports/results/pytest_junit.xml${slow ? ' and pytest_junit_slow.xml' : ''}, reports/api_demo/summary.json`,
      notes: [
        `Testing evidence is saved, not claimed. The fast pytest suite, stored as JUnit XML, has ${fmtInt(j.tests)} tests: ${jPassed === j.tests && j.tests > 0 ? 'all passed' : `${fmtInt(jPassed)} passed`}${j.failures ? `, ${plural(j.failures, 'failure')}` : ''}${j.errors ? `, ${plural(j.errors, 'error')}` : ''}${j.skipped ? `, ${fmtInt(j.skipped)} skipped` : ''}${broken.length ? ` (${broken.join(', ')})` : ''}, in ${Math.round(j.seconds)} seconds; ${mods[0][0]} has the most tests, then ${listJoin(mods.slice(1, 4).map(([m]) => m))}.${slow ? ` The slow suite - CTGAN and TVAE fits, a full simulator run and end-to-end API and evaluation runs - passed ${fmtInt(slowPassed)} of ${fmtInt(slow.tests)}.` : ''}`,
        `The API demo script calls the application in-process against the real models and stores one evidence file per call: ${passed} of ${total} steps passed, covering ${covered.length} of ${ctx.endpoints.length} endpoints${extraPaths.length ? ` plus ${extraPaths.join(', ')}` : ''}, ${errorSteps.length} error cases (${listJoin(errorCodes.map(String))}), ${scenOk} of ${scenarioSteps.length} scenarios and ${reproduced ? 'a passing' : 'a failing'} same-seed reproducibility check.`,
        `Now the live demo, about three minutes (docs/DEMO_SCRIPT.md section 4): the Ready pill in /ui; a high-occupancy request with seed 42 and Evaluate ticked - tiles, Validation, Quality; /validate flagging a broken record; an impossible filter refused before sampling; a saved dataset downloaded as CSV. If the service fails, use the fallback of section 5 (reports/api_demo/summary.md). Then hand over to Member C.`,
      ].join('\n\n'),
    });
    const lw = 6.15;
    slide.addChart(pres.ChartType.bar, [{ name: 'Tests', labels: mods.map(([m]) => m), values: mods.map(([, n]) => n) }], Object.assign(chartBase(), {
      x: MX, y: TOP, w: lw, h: slow ? 4.05 : 4.4, barDir: 'bar', chartColors: [HEX.accent1], catAxisOrientation: 'maxMin', valAxisHidden: true,
      valGridLine: { style: 'none' }, showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0', showLegend: false,
      showTitle: true, title: 'pytest cases per test module (fast suite)', catAxisLabelFontFace: CODE_FONT, barGapWidthPct: 35,
      valAxisMaxVal: Math.ceil((mods[0][1] * 1.18) / 10) * 10,
    }));
    // non-zero failure and error counts in red
    const count = (n, label) => ({ text: `${n} ${label}`, options: n ? { color: C.red, bold: true } : { color: HEX.muted } });
    K.text(slide, [
      { text: `${fmtInt(jPassed)} passed`, options: { bold: true, color: C.teal } },
      { text: ' · ', options: { color: HEX.muted } },
      count(j.failures, 'failed'),
      { text: ' · ', options: { color: HEX.muted } },
      count(j.errors, 'errors'),
      { text: ` · ${j.skipped} skipped · ${Math.round(j.seconds)} s`, options: { color: HEX.muted } },
    ], { x: MX, y: TOP + (slow ? 4.12 : 4.5), w: lw, h: 0.36, fontSize: 15, align: 'center' });
    if (slow) {
      K.text(slide, [
        { text: `Slow suite (pytest -m slow): ${fmtInt(slowPassed)}/${fmtInt(slow.tests)} passed`, options: { bold: true, color: slow.failures + slow.errors ? C.red : C.ink } },
        { text: ` · ${Math.round(slow.seconds)} s`, options: { color: HEX.muted } },
      ], { x: MX, y: TOP + 4.52, w: lw, h: 0.34, fontSize: 14, align: 'center' });
    }
    const rx = MX + lw + 0.4;
    const rw = W - MX - rx;
    K.box(slide, { x: rx, y: TOP, w: rw, h: 4.9, radius: 0.1, fill: C.tint, name: 'API demo' });
    K.text(slide, 'API DEMO · scripts/05_api_demo.py', { x: rx + 0.3, y: TOP + 0.2, w: rw - 0.6, h: 0.28, fontSize: 12, bold: true, color: C.teal, charSpacing: 1 });
    K.stat(slide, rx + 0.3, TOP + 0.45, rw - 0.6, `${passed}/${total}`, 'steps passed, one evidence file per call', { size: 40, valueH: 0.7 });
    const facts = [
      ['check', `${covered.length}/${ctx.endpoints.length} listed endpoints called in-process${extraPaths.length ? ` (+${NBSP}${extraPaths.join(', ')})` : ''}`],
      ['alert', `${errorSteps.length} error cases (${errorCodes.join(' / ')}) return uniform JSON errors`],
      ['tiles', `${scenOk}/${scenarioSteps.length} scenarios generated, all hard rules passed`],
      ['engine', reproduced ? 'Same seed → identical records' : 'Reproducibility check not passed'],
    ];
    facts.forEach(([icon, t], i) => {
      const y = TOP + 1.85 + i * 0.74;
      K.icon(slide, icon, rx + 0.3, y, 0.48);
      K.text(slide, t, { x: rx + 0.95, y, w: rw - 1.2, h: 0.5, fontSize: 15, valign: 'middle' });
    });
  }

  // ---------------------------------------------------------------- 19. Limitations and future work
  {
    const failedWD = opWD.failed_checks || [];
    const failedPS = opPS.failed_checks || [];
    const failedAll = [...failedWD, ...failedPS].map(checkLabel);
    // utility families below the no-learning floor: the same list as slide 16
    const ug = ctx.utilityGaps;
    const floorWord = ug.every((g) => g.level) ? 'level with' : ug.some((g) => g.level) ? 'at or below' : 'below';
    const utilShort = ug.length ? `${listJoin(ug.map((g) => `${g.t === 'ward_daily' ? 'ward-day' : 'stay'} ${g.k === 'ml_utility' ? 'ML' : 'optimisation'} utility`))} ${floorWord} the floor` : '';
    const utilNotes = ug.length ? `${listJoin(ug.map((g) => `${g.k === 'ml_utility' ? 'machine-learning' : 'optimisation'} utility on ${g.t}`))} ${floorWord} the no-learning floor` : '';
    // remaining measured gaps of the default models, most serious first (all computed from the results)
    const beyond = TABLES.map((t) => [t, ctx.opGaps[t].beyondNoise.map(checkLabel)]).filter(([, l]) => l.length);
    const weak = ctx.weakestScenarios;
    const checksWord = (n) => `${word(n)} check${n === 1 ? '' : 's'}`;
    const cardItems = [
      ...[[failedWD, 'ward_daily'], [failedPS, 'patient_stays']].filter(([f]) => f.length).map(([f, t]) => `${checksWord(f.length)} failing on ${t}`),
      ...(utilShort ? [utilShort] : []),
      ...ctx.componentGaps.map((g) => `${g.t} ${g.f} ${f3(g.m)} vs floor ${f3(g.floor)}`),
      ...beyond.map(([t, l]) => `${checksWord(l.length)} on ${t} beyond real-data noise`),
    ];
    if (!cardItems.length && weak.length) cardItems.push(`weakest scenarios ${listJoin(weak.map((s) => `${human(s.name)} (${f2(s.score)})`))}`);
    const gapCard = cardItems.length ? `${cardItems.slice(0, 2).join('; ')}.` : 'none measured.';
    const gapNotes = [
      failedAll.length ? `${listJoin(failedAll)} ${failedAll.length === 1 ? 'fails' : 'fail'}` : '',
      utilNotes,
      ...ctx.componentGaps.map((g) => `${g.t} ${g.f} is ${f3(g.m)}, ${g.floor - g.m < 0.01 ? 'just ' : ''}below the no-learning floor of ${f3(g.floor)}`),
      ...beyond.map(([t, l]) => `${checksWord(l.length)} on ${t} - ${listJoin(l)} - pass but lie beyond real-data noise`),
      weak.length ? `${ctx.weakFiltered ? 'the filtered scenarios' : 'the scenarios'} ${listJoin(weak.map((s) => s.name))} score lowest` : '',
    ].filter(Boolean);
    let target;
    if (failedAll.length) target = { card: `for the failing checks, e.g. ${listJoin(failedAll.slice(0, 2))}.`, notes: `the failing checks (${shortList(failedAll)})` };
    else if (beyond.length) {
      const l = beyond.flatMap(([, x]) => x);
      target = { card: `for the checks beyond real-data noise, e.g. ${listJoin(l.slice(0, 2))}.`, notes: 'the checks beyond real-data noise' };
    } else if (weak.length) target = { card: `for the weakest scenarios, e.g. ${listJoin(weak.map((s) => human(s.name)))}.`, notes: 'the weakest scenarios' };
    else target = { card: 'for the weakest scenario scores.', notes: 'the weakest scenarios' };
    // how far the numbers rest on one evaluation seed: the A/B re-scores of the ward_daily default
    // on further seeds when available, otherwise the single-seed caveat
    const ab = ctx.ab;
    let seedNote = '';
    if (ab) {
      const all = { 1: 'it', 2: 'both' }[ab.extraSeeds] || 'all of them';
      const where = ab.failingSeeds === ab.extraSeeds ? all : `${word(ab.failingSeeds)} of them`;
      const names = ab.failedChecks.map((c) => `the ${checkLabel(c)}`);
      seedNote = ` Most numbers rest on ${ctx.evalSeeds.length === 1 ? `evaluation seed ${ctx.evalSeeds[0]} alone` : 'a single evaluation run'}; re-scored on ${word(ab.extraSeeds)} more seed${ab.extraSeeds === 1 ? '' : 's'}, the ward_daily default ${ab.failingSeeds
        ? `fails ${ab.maxFailing === 1 ? 'one check' : `up to ${word(ab.maxFailing)} checks`} on ${where}${names.length ? ` - ${listJoin(names, 'or')}` : ''}`
        : `passes every check on ${all}`}.`;
    } else if (ctx.evalSeeds.length === 1) {
      seedNote = ` Every number comes from one evaluation seed (${ctx.evalSeeds[0]}), so a check close to its threshold can flip on another seed; DESIGN section 7.2 shows this for ward_daily.`;
    }
    const slide = contentSlide({
      member: 'C', section: 'close', kicker: 'Closing · limitations', title: 'Limitations and future work', seconds: 75,
      notes: [
        `Thank you. To finish, the limits. The reference data is simulated; a real deployment would retrain on a facility extract with the same schema. Rows are conditionally independent, so calendar mode has no day-to-day census continuity, and the two tables are synthesised independently. ${ctx.aggMirror ? 'The allocation strata reproduce training frequencies exactly, an aggregate disclosure.' : 'The allocation strata follow the training frequencies, an aggregate disclosure.'}`,
        `The evaluation also leaves measured gaps${gapNotes.length ? `: ${gapNotes.join('; ')}` : ', though none below its threshold or the floor'}.${seedNote}`,
        `Future work follows: a sequential model such as PARSynthesizer for census continuity, a multi-table model linking stays to ward-days, which needs a single-column ward-day key for HMASynthesizer, noise on the strata weights or a differentially private synthesizer, targeted constraints for ${target.notes}, and a pilot with a hospital operations team.${ctx.quick ? ' The numbers in this deck come from the quick run; the full run trains CTGAN and TVAE properly and evaluates more rows.' : ''}`,
      ].join('\n\n'),
    });
    const lim = [
      ['Simulated reference data', 'Retrain on a real facility extract with the same schema.'],
      ['No day-to-day continuity', 'Rows are conditionally independent; calendar mode has no census carry-over.'],
      ['Tables synthesised independently', 'No linkage between stays and ward-days.'],
      ['Aggregate disclosure', ctx.aggMirror ? 'Allocation strata copy the training frequencies exactly.' : 'Allocation strata follow the training frequencies.'],
      ['Remaining quality gaps', gapCard],
    ];
    const fut = [
      ['Sequential model', 'e.g. SDV PARSynthesizer for census continuity.'],
      ['Multi-table model', 'link stays to ward-days (HMASynthesizer needs a single-column ward-day key).'],
      ['Stronger privacy', 'Noise on strata weights or a differentially private synthesizer.'],
      ['Targeted constraints', target.card],
      [ctx.quick ? 'Full-run numbers' : 'Real-data pilot', ctx.quick ? 'Use the full training and evaluation run for final figures.' : 'Validate the generator with a hospital operations team.'],
    ];
    const cw = (CW - 0.4) / 2;
    [[lim, 'Limitations', 'alert', C.red], [fut, 'Future work', 'future', C.teal]].forEach(([items, head, icon, color], k) => {
      const x = MX + k * (cw + 0.4);
      K.icon(slide, icon, x, TOP, 0.56, color);
      K.text(slide, head, { x: x + 0.72, y: TOP + 0.08, w: cw - 0.8, h: 0.4, fontSize: 20, bold: true });
      items.forEach(([a, b], i) => {
        const y = TOP + 0.76 + i * 0.87;
        K.box(slide, { x, y, w: cw, h: 0.8, radius: 0.08, fill: k === 0 ? 'F9EFEC' : C.tint, name: `${head} ${i + 1}` });
        K.text(slide, [{ text: `${a}: `, options: { bold: true } }, { text: b, options: { color: HEX.muted } }], { x: x + 0.2, y, w: cw - 0.4, h: 0.8, fontSize: 14, valign: 'middle' });
      });
    });
  }

  // ---------------------------------------------------------------- 20. Team and Q&A
  {
    const slide = pres.addSlide({ masterName: 'BS_TITLE_DARK', sectionTitle: ensureSection('close') });
    K.text(slide, 'CLOSING', { x: 0.8, y: 0.62, w: 6, h: 0.32, fontSize: 14, bold: true, color: HEX.lt2, charSpacing: 1.5 });
    slide.addText('Thank you — questions?', { placeholder: 'title' });
    const team = [
      ['A', 'Week 1', 'Data preparation', 'cleaning, rules, scenarios'],
      ['B', 'Week 2', 'SDV pipeline', `${nExperiments} experiments, constraints`],
      ['C', 'Week 3', 'Evaluation', 'metrics, references, ablation'],
      ['D', 'Week 4', 'API, testing and demo', `FastAPI, ${fmtInt(d.junit.tests)}${d.junitSlow ? ` + ${fmtInt(d.junitSlow.tests)}` : ''} tests, demo`],
    ];
    const card = (m, wk, role, what, x, y, cw, ch) => {
      K.box(slide, { x, y, w: cw, h: ch, radius: 0.12, fill: C.bg, transparency: 90, name: `Speaker ${m}` });
      K.circle(slide, x + 0.25, y + 0.22, 0.56, C.amber);
      K.text(slide, m, { x: x + 0.25, y: y + 0.22, w: 0.56, h: 0.56, fontSize: 20, bold: true, color: C.bg, align: 'center', valign: 'middle' });
      K.text(slide, MEMBERS[m], { x: x + 0.96, y: y + 0.2, w: cw - 1.1, h: 0.34, fontSize: 17, bold: true, color: C.bg });
      K.text(slide, wk, { x: x + 0.96, y: y + 0.52, w: cw - 1.1, h: 0.3, fontSize: 14, color: HEX.lt2 });
      K.text(slide, role, { x: x + 0.25, y: y + 0.92, w: cw - 0.4, h: 0.32, fontSize: 15, bold: true, color: C.bg });
      K.text(slide, what, { x: x + 0.25, y: y + 1.26, w: cw - 0.4, h: ch - 1.34, fontSize: 14, color: HEX.lt2 });
    };
    if (d.uiScreenshot) {
      // 2 x 2 team grid on the left, the live demo UI on the right
      const cw = 3.05;
      const ch = 1.8;
      team.forEach(([m, wk, role, what], i) => card(m, wk, role, what, 0.8 + (i % 2) * (cw + 0.22), 2.45 + Math.floor(i / 2) * (ch + 0.22), cw, ch));
      const sx = 0.8 + 2 * cw + 0.22 + 0.45;
      const sw = 12.53 - sx;
      const sh = Math.min(3.72, (sw * d.uiScreenshot.size.h) / d.uiScreenshot.size.w);
      const sw2 = (sh * d.uiScreenshot.size.w) / d.uiScreenshot.size.h;
      slide.addImage({ path: d.uiScreenshot.path, x: sx + (sw - sw2) / 2, y: 2.45, w: sw2, h: sh, altText: 'Screenshot of the BedSynth AI demo UI at /ui after a high-occupancy ward_daily request with quality evaluation: request form, result summary cards and the first records.' });
      K.text(slide, 'Demo UI at /ui after a high-occupancy request (screenshot)', { x: sx, y: 2.45 + sh + 0.08, w: sw, h: 0.28, fontSize: 12, italic: true, color: HEX.lt2, align: 'center' });
      K.text(slide, [
        { text: 'python scripts/04_serve_api.py', options: { fontFace: CODE_FONT, color: C.bg } },
        { text: '  →  /ui', options: { color: HEX.lt2 } },
      ], { x: sx, y: 2.45 + sh + 0.42, w: sw, h: 0.34, fontSize: 14, align: 'center' });
    } else {
      const cw = (11.73 - 3 * 0.3) / 4;
      team.forEach(([m, wk, role, what], i) => card(m, wk, role, what, 0.8 + i * (cw + 0.3), 2.55, cw, 2.2));
      K.text(slide, [
        { text: 'Try it: ', options: { bold: true, color: C.bg } },
        { text: 'python scripts/04_serve_api.py', options: { fontFace: CODE_FONT, color: C.bg } },
        { text: '  →  open /ui or /docs', options: { color: HEX.lt2 } },
      ], { x: 0.8, y: 5.2, w: 11.7, h: 0.4, fontSize: 16 });
    }
    K.text(slide, 'Read more: README.md (Results) · docs/DESIGN.md · reports/week1_data_preparation.md … week4_api_system.md', { x: 0.8, y: 6.62, w: 11.7, h: 0.36, fontSize: 14, color: HEX.lt2 });
    slide.addNotes(named(`Suggested speaker: Member C, then all members for questions. Target time: about 45 seconds.\n\n${[
      `To close: BedSynth AI turns an operational extract, here a simulated one, into SDV models that generate realistic, rule-valid data for ${ctx.nScenarios} scenarios on demand, scores every model against real-data and no-learning references, and serves the result through a tested FastAPI service.`,
      `Each of us takes questions on our own part: Member A the data and cleaning, Member B SDV and the constraints, Member C the evaluation and limitations, Member D the API and testing. The full numbers are in the README Results section and reports/week3_evaluation.md. If the live demo was skipped, it can be shown now. Thank you.`,
    ].join('\n\n')}`));
  }
}

// ------------------------------------------------------------------------------------------
// Post-processing: write the theme colours (pptxgenjs writes Office's palette) and mend
// pptxgenjs quirks: crossesAt="autoZero", the undeclared series-axis id it adds to every 2-D
// bar/line chart, the order of the notes-master list in presentation.xml, content-type
// overrides for slide masters that do not exist, and folder entries in the ZIP.
// ------------------------------------------------------------------------------------------
async function applyTheme(file, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = 'ppt/theme/theme1.xml';
  if (!zip.file(part)) fail(`${file}: no ${part}`);
  const slots = ['dk1', 'lt1', 'dk2', 'lt2', 'accent1', 'accent2', 'accent3', 'accent4', 'accent5', 'accent6', 'hlink', 'folHlink'];
  const name = theme.name.replace(/[&<>"]/g, '');
  const scheme = `<a:clrScheme name="${name}">${slots.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join('')}</a:clrScheme>`;
  let xml = await zip.file(part).async('string');
  xml = xml.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${name}"`);
  if (!xml.includes(scheme)) fail('theme1.xml has no colour scheme to replace');
  zip.file(part, xml);
  for (const entry of Object.keys(zip.files)) {
    if (!entry.endsWith('.xml')) continue;
    let content = await zip.file(entry).async('string');
    const original = content;
    const bad = content.match(/<a:srgbClr val="((?![0-9A-Fa-f]{6}")[^"]*)"/);
    if (bad) fail(`${entry}: colour "${bad[1]}" is not a six-digit hex value (a scheme colour in a hex-only option)`);
    content = content.split('<c:crossesAt val="autoZero"/>').join('<c:crosses val="autoZero"/>');
    if (/^ppt\/charts\/chart\d+\.xml$/.test(entry)) {
      // drop axis ids that no axis element declares (2-D charts only have a category and a value axis)
      const declared = new Set([...content.matchAll(/<c:(?:catAx|valAx|dateAx|serAx)>\s*<c:axId val="(\d+)"\/>/g)].map((m) => m[1]));
      content = content.replace(/<c:axId val="(\d+)"\/>/g, (tag, id) => (declared.has(id) || !declared.size ? tag : ''));
    }
    if (content !== original) zip.file(entry, content);
  }
  // schema order of <p:presentation>: sldMasterIdLst, notesMasterIdLst, handoutMasterIdLst, sldIdLst
  // (pptxgenjs writes the notes master list after the slide list)
  const presPart = 'ppt/presentation.xml';
  const presXml = await zip.file(presPart).async('string');
  const notesList = (presXml.match(/<p:notesMasterIdLst>[\s\S]*?<\/p:notesMasterIdLst>/) || [])[0];
  if (notesList && presXml.indexOf(notesList) > presXml.indexOf('<p:sldIdLst>') && presXml.includes('</p:sldMasterIdLst>')) {
    zip.file(presPart, presXml.replace(notesList, '').replace('</p:sldMasterIdLst>', `</p:sldMasterIdLst>${notesList}`));
  }
  const ctPart = '[Content_Types].xml';
  const ct = await zip.file(ctPart).async('string');
  zip.file(ctPart, ct.replace(/<Override PartName="\/([^"]+)"[^>]*\/>/g, (tag, part) => (zip.file(part) ? tag : '')));
  for (const entry of Object.keys(zip.files)) if (zip.files[entry].dir) delete zip.files[entry];
  fs.writeFileSync(file, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const data = loadInputs(cfg);
  const ctx = buildContext(data);
  const pres = createPresentation();
  buildSlides(pres, data, ctx);
  fs.mkdirSync(path.dirname(cfg.out), { recursive: true });
  await pres.writeFile({ fileName: cfg.out });
  await applyTheme(cfg.out, THEME);
  const shown = path.relative(process.cwd(), cfg.out) || cfg.out;
  console.log(`Wrote ${shown}: ${pres.slides.length} slides${ctx.quick ? ' (quick-run results)' : ''}`);
}

main().catch((err) => {
  if (err instanceof InputError) {
    console.error(`build_deck: ERROR: ${err.message}`);
  } else {
    console.error(`build_deck: unexpected error: ${err && err.stack ? err.stack : err}`);
  }
  process.exit(1);
});
