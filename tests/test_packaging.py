"""Tests for ``scripts/package_submission.py`` (submission zip and identifier scan) and the JUnit
host-name stripping of ``scripts/05_api_demo.py``. Everything runs in temporary directories."""
from __future__ import annotations

import importlib.util
import io
import json
import pickle
import socket
import sys
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_script(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "scripts" / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module           # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def packaging():
    return _load_script("package_submission.py", "bedsynth_package_submission_script")


def _write(path: Path, content: str | bytes = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))   # no newline translation
    return path


def _deck(path: Path, slide_xml: str) -> Path:
    """A minimal zip container shaped like a .pptx (XML parts are compressed, so raw bytes hide them)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("ppt/slides/slide1.xml", slide_xml)
    return path


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A small project tree with deliverables, build folders and data that must stay out."""
    root = tmp_path / "project"
    for name in ("README.md", "requirements.txt", "pyproject.toml", "bedsynth/__init__.py", "bedsynth/api/main.py",
                 "scripts/run_pipeline.py", "config/settings.yaml", "tests/test_x.py", "demo/requests.http",
                 "docs/DESIGN.md", "docs/presentation/build_deck.js", "reports/week1_data_preparation.md",
                 "reports/results/model_comparison.json", "data/processed/ward_daily_train.csv",
                 "models/registry.json", "data/synthetic/api_demo/ds-0123456789ab/data.csv"):
        _write(root / name, f"content of {name}\n")
    _write(root / "models/ward_daily/gc_tuned/synthesizer.pkl", pickle.dumps({"source": "data/processed/x.csv"}))
    _deck(root / "docs/presentation/BedSynth_AI.pptx", "<p:sld><a:t>BedSynth AI</a:t></p:sld>")
    for name in ("bedsynth/__pycache__/x.cpython-313.pyc", "tests/.pytest_cache/v/cache", ".venv/lib/site.py",
                 "docs/presentation/node_modules/pptxgenjs/index.js", "dist/old_submission.zip",
                 "data/raw/ward_daily_raw.csv", "data/synthetic/api/ds-ffffffffffff/data.csv",
                 ".pytest_cache/README.md", "scripts/stale.pyc"):
        _write(root / name, "build output\n")
    return root


def _relative(files: list[Path], root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in files}


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def test_collect_files_takes_deliverables_and_skips_build_and_raw_folders(packaging, project):
    files = _relative(packaging.collect_files(project), project)
    assert {"README.md", "requirements.txt", "pyproject.toml", "bedsynth/api/main.py", "scripts/run_pipeline.py",
            "config/settings.yaml", "tests/test_x.py", "demo/requests.http", "docs/presentation/BedSynth_AI.pptx",
            "docs/presentation/build_deck.js", "reports/results/model_comparison.json",
            "data/processed/ward_daily_train.csv", "models/ward_daily/gc_tuned/synthesizer.pkl",
            "data/synthetic/api_demo/ds-0123456789ab/data.csv"} <= files
    for excluded in ("__pycache__", ".pytest_cache", ".venv", "node_modules", "dist/", "data/raw/",
                     "data/synthetic/api/", ".pyc"):
        assert not any(excluded in name for name in files), excluded
    with_raw = _relative(packaging.collect_files(project, include_raw=True), project)
    assert with_raw - files == {"data/raw/ward_daily_raw.csv"}
    skipped = packaging.collect_files(project, skip=[project / "README.md"])
    assert "README.md" not in _relative(skipped, project)


# ---------------------------------------------------------------------------
# Identifier patterns and scanning
# ---------------------------------------------------------------------------
def test_home_pattern_matches_every_separator_style(packaging):
    patterns = packaging.leak_patterns(home=PureWindowsPath(r"C:\Users\alice"), hostname="BUILD-PC-42", user="alice")
    home = patterns["home directory"]
    for text in (rb"C:\Users\alice\project\x.csv", b"C:/Users/alice/project", rb'{"p": "C:\\Users\\alice\\x"}',
                 b"/c/Users/alice/x", rb"c:\users\ALICE", rb"D:\Users\alice\x"):
        assert home.search(text), text
    for text in (rb"C:\Users\alicex\y", b"~/AppData/Local/Temp/x", b"Users alice", b"data/processed/x.csv"):
        assert not home.search(text), text
    assert packaging.leak_patterns(home=PurePosixPath("/home/bob"), hostname="", user="")["home directory"].search(
        b"/home/bob/project")
    assert patterns["user name"].search(b"~/AppData/Local/Temp/pytest-of-alice/pytest-3/x")
    host = patterns["host name"]
    assert host.search(b'hostname="build-pc-42"') and not host.search(b"BUILD-PC-420") and not host.search(b"XBUILD-PC-42")
    assert "host name" not in packaging.leak_patterns(home=PurePosixPath("/home/bob"), hostname="pc", user="bob")
    assert "host name" not in packaging.leak_patterns(home=PurePosixPath("/home/bob"), hostname="localhost", user="bob")


def test_scan_reads_pickles_utf16_and_the_xml_inside_office_files(packaging, tmp_path):
    patterns = packaging.leak_patterns(home=PureWindowsPath(r"C:\Users\alice"), hostname="BUILD-PC-42", user="alice")
    secret = r"C:\Users\alice\Desktop\genaiproject\data\processed\ward_daily_train.csv"
    pickled = _write(tmp_path / "synthesizer.pkl", pickle.dumps({"_annotations": {"source": secret}}))
    utf16 = _write(tmp_path / "notes.bin", ("\x00prefix " + secret).encode("utf-16-le"))
    filler = "<a:t>BedSynth AI ward occupancy</a:t>" * 50          # compressible: the XML is deflated
    deck = _deck(tmp_path / "deck.pptx", f"<p:sld>{filler}<a:t>{secret}</a:t><a:t>BUILD-PC-42</a:t>{filler}</p:sld>")
    clean = _write(tmp_path / "clean.json", json.dumps({"source": "data/processed/ward_daily_train.csv"}))

    assert {h.kind for h in packaging.scan_file(pickled, "synthesizer.pkl", patterns)} == {"home directory"}
    assert packaging.scan_file(utf16, "notes.bin", patterns)
    assert packaging.scan_bytes(deck.read_bytes(), "deck.pptx", patterns) == []   # invisible in the raw bytes
    deck_hits = packaging.scan_file(deck, "deck.pptx", patterns)
    assert {(h.location, h.kind) for h in deck_hits} == {("deck.pptx!ppt/slides/slide1.xml", "home directory"),
                                                         ("deck.pptx!ppt/slides/slide1.xml", "host name")}
    assert all("alice" not in h.context.lower() and "build-pc-42" not in h.context.lower() for h in deck_hits)
    assert packaging.scan_file(clean, "clean.json", patterns) == []


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def test_dry_run_lists_files_and_total_and_writes_nothing(packaging, project, capsys):
    assert packaging.main(["--root", str(project), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "README.md" in out and "models/ward_daily/gc_tuned/synthesizer.pkl" in out
    assert "files," in out and "before compression" in out and "Dry run: nothing written." in out
    assert "data/raw/" not in out and "node_modules" not in out
    assert not (project / "dist" / "BedSynth_AI_submission.zip").exists()


def test_build_writes_one_folder_zip_without_build_outputs(packaging, project, tmp_path):
    out = tmp_path / "out" / "submission.zip"
    assert packaging.main(["--root", str(project), "--out", str(out)]) == 0
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        assert archive.read("BedSynth_AI/README.md") == b"content of README.md\n"
    assert all(name.startswith("BedSynth_AI/") for name in names)
    assert "BedSynth_AI/docs/presentation/BedSynth_AI.pptx" in names
    assert not any(part in name for name in names for part in ("__pycache__", "node_modules", ".venv", "data/raw"))
    assert not list(out.parent.glob("*.partial"))
    # default location: dist/ under the project root, never packaged itself on the next run
    assert packaging.main(["--root", str(project)]) == 0
    default = project / "dist" / "BedSynth_AI_submission.zip"
    assert default.is_file()
    assert packaging.main(["--root", str(project), "--out", str(out)]) == 0
    with zipfile.ZipFile(out) as archive:
        assert not any(name.endswith(".zip") for name in archive.namelist())


def test_build_aborts_on_this_machines_home_path_and_host_name(packaging, project, tmp_path, capsys):
    patterns = packaging.leak_patterns()            # computed for the machine running the tests
    if "home directory" not in patterns:
        pytest.skip("no home directory to look for on this machine")
    _write(project / "models" / "registry.json", json.dumps({"path": str(Path.home() / "project" / "models")}))
    host = socket.gethostname()
    if "host name" in patterns:
        _write(project / "reports" / "results" / "pytest_junit.xml", f'<testsuite hostname="{host}" tests="1"/>')
    out = tmp_path / "submission.zip"
    assert packaging.main(["--root", str(project), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert not out.exists() and "Aborted: nothing written" in err
    assert "models/registry.json: home directory" in err
    if "host name" in patterns:
        assert "reports/results/pytest_junit.xml: host name" in err
    assert str(Path.home()) not in err and (len(host) < 3 or host not in err)   # the listing masks the identifiers
    assert packaging.main(["--root", str(project), "--out", str(out), "--allow-leaks"]) == 0 and out.is_file()


# ---------------------------------------------------------------------------
# scripts/05_api_demo.py: the JUnit evidence carries no host name
# ---------------------------------------------------------------------------
def test_api_demo_strips_the_junit_host_name_and_nothing_else(tmp_path):
    demo = _load_script("05_api_demo.py", "bedsynth_api_demo_script_packaging")
    junit = tmp_path / "pytest_junit.xml"
    original = ('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests"><testsuite name="pytest" '
                'errors="0" failures="1" skipped="1" tests="3" time="2.5" timestamp="2026-11-15T00:00:00" '
                'hostname="BUILD-PC-42"><testcase classname="tests.test_a" name="test_x" time="1" />'
                '<testcase classname="tests.test_a" name="test_y" time="1"><failure message="hostname=x">'
                'hostname="kept"</failure></testcase><testcase classname="tests.test_a" name="test_z" time="0.5">'
                '<skipped message="slow" /></testcase></testsuite></testsuites>')
    junit.write_text(original, encoding="utf-8")
    assert demo.strip_junit_hostname(junit) is True
    assert junit.read_text(encoding="utf-8") == original.replace(' hostname="BUILD-PC-42"', "")
    assert demo.strip_junit_hostname(junit) is False
    assert demo.junit_totals(junit) == {"tests": 3, "failures": 1, "errors": 0, "skipped": 1, "passed": 1,
                                        "seconds": 2.5}
    lines = demo._test_runs_markdown([{"suite": "slow", "command": 'python -m pytest -q -m "slow"', "ok": True,
                                       "exit_code": 0, "junit_xml": "reports/results/pytest_junit_slow.xml",
                                       "totals": demo.junit_totals(junit)}])
    assert "## Test suites" in lines and any("| slow |" in line and "pytest_junit_slow.xml" in line for line in lines)
    args = demo.parse_args(["--run-slow-tests"])
    assert args.run_slow_tests and not args.run_tests


def test_api_demo_redacts_machine_identifiers_in_junit_tracebacks(tmp_path, monkeypatch):
    """A failing test's traceback carries pytest's temp path (home directory, pytest-of-<user>)."""
    demo = _load_script("05_api_demo.py", "bedsynth_api_demo_script_identifiers")
    fake_home = Path(r"C:\Users\Some One") if sys.platform == "win32" else Path("/home/someone")
    monkeypatch.setattr(demo.Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(demo.getpass, "getuser", lambda: "someone")
    monkeypatch.setattr(demo.platform, "node", lambda: "BUILD-PC-42")
    home = str(fake_home)
    repr_home = home.replace("\\", "\\\\")                  # how repr() shows a Windows path
    sep = "\\" if sys.platform == "win32" else "/"
    junit = tmp_path / "pytest_junit.xml"
    original = ('<testsuite name="pytest" tests="1" failures="1"><testcase name="t"><failure message="boom">'
                "models_dir='" + repr_home + "\\\\AppData\\\\pytest-of-someone\\\\pytest-1'\r\n"
                "File " + home + sep + "proj" + sep + "x.py on BUILD-PC-42</failure></testcase></testsuite>")
    junit.write_bytes(original.encode("utf-8"))
    assert demo.scrub_junit_identifiers(junit) >= 3
    text = junit.read_bytes().decode("utf-8")
    assert home not in text and repr_home not in text and "someone" not in text and "BUILD-PC-42" not in text
    assert "pytest-of-USER" in text and "<host>" in text and "\r\n" in text     # line endings are kept
    assert demo.scrub_junit_identifiers(junit) == 0                            # idempotent
    assert demo.scrub_junit_identifiers(tmp_path / "missing.xml") == 0


# ---------------------------------------------------------------------------
# scripts/build_vercel_bundle.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vercel_bundle():
    return _load_script("build_vercel_bundle.py", "bedsynth_build_vercel_bundle_script")


def test_vercel_registry_drops_torch_models_and_keeps_the_default(vercel_bundle, tmp_path):
    registry = {
        table: {"default": "gc_tuned", "models": {m: {"experiment_id": m} for m in ("gc_tuned", "ctgan", "tvae")}}
        for table in ("ward_daily", "patient_stays")
    }
    path = _write(tmp_path / "registry.json", json.dumps(registry))
    pruned = vercel_bundle._filtered_registry(path)
    assert {table: list(entry["models"]) for table, entry in pruned.items()} == {
        "ward_daily": ["gc_tuned"], "patient_stays": ["gc_tuned"]}
    assert all(entry["default"] == "gc_tuned" for entry in pruned.values())


def test_vercel_registry_refuses_a_torch_default(vercel_bundle, tmp_path):
    path = _write(tmp_path / "registry.json", json.dumps({"ward_daily": {"default": "ctgan", "models": {"ctgan": {}}}}))
    with pytest.raises(SystemExit, match="PyTorch"):
        vercel_bundle._filtered_registry(path)


def test_vercel_bundle_installs_no_torch_and_routes_everything_to_the_app(vercel_bundle):
    requirements = [line.split("==")[0].strip().lower() for line in vercel_bundle.REQUIREMENTS.splitlines()
                    if line.strip() and not line.startswith("#")]
    assert not {"torch", "ctgan", "deepecho", "sdv"} & set(requirements)
    function = vercel_bundle.VERCEL_CONFIG["functions"]["api/index.py"]
    assert "models" in function["includeFiles"] and "sdv" in function["includeFiles"]
    assert vercel_bundle.VERCEL_CONFIG["rewrites"] == [{"source": "/(.*)", "destination": "/api/index?__path=/$1"}]
    assert vercel_bundle.PUBLIC_API_LIMITS["max_rows"] <= 2000
