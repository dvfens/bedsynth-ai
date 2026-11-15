"""Build the BedSynth AI submission zip: the deliverables only, checked for machine-identifying data.

Usage (from the project root)::

    .venv/Scripts/python.exe scripts/package_submission.py --dry-run     # list the files and the total size
    .venv/Scripts/python.exe scripts/package_submission.py               # writes dist/BedSynth_AI_submission.zip
    .venv/Scripts/python.exe scripts/package_submission.py --out handin/BedSynth_AI.zip --include-raw

Contents (inside one top-level ``BedSynth_AI/`` folder): ``README.md``, ``requirements.txt``,
``pyproject.toml``, ``bedsynth/``, ``scripts/``, ``config/``, ``tests/``, ``demo/``, ``docs/``
(without ``docs/presentation/node_modules``), ``reports/``, ``data/processed/``, ``models/`` and
``data/synthetic/api_demo/`` (the datasets of the API demo). Never included: ``.venv``,
``__pycache__``, ``.pytest_cache``, ``dist/`` and compiled ``*.pyc`` files; ``data/raw/`` (the
simulated source extract) only with ``--include-raw``.

Before anything is written, every included file is scanned for the current user's home-directory
path (any separator style, JSON-escaped included, drive letter optional), the user name inside a
pytest temporary path (``pytest-of-<user>``) and the machine's host name - all determined when the
script runs, never stored in it. Text files, pickles and other binary files are scanned byte by
byte (binary files also as UTF-16 text), and zip containers (``.pptx``, ``.docx``, ``.xlsx``,
``.zip``) member by member, so the XML of the slide deck is covered. Any hit aborts the build and
lists file, kind and context (with the identifier masked); ``--allow-leaks`` writes the zip anyway.
``--dry-run`` lists the files, their total size and the scan result without writing anything.

Exit codes: 0 written (or dry run without findings), 1 identifying data found (nothing written
unless ``--allow-leaks``), 2 nothing to package.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import socket
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("dist") / "BedSynth_AI_submission.zip"
ARCHIVE_FOLDER = "BedSynth_AI"
INCLUDED: tuple[str, ...] = (
    "README.md", "requirements.txt", "pyproject.toml", "bedsynth", "scripts", "config", "tests", "demo", "docs",
    "reports", "data/processed", "models", "data/synthetic/api_demo",
)
RAW_DATA = "data/raw"
#: Project-relative locations never packaged (whatever ``INCLUDED`` says).
EXCLUDED_PATHS: tuple[str, ...] = (".venv", "dist", "docs/presentation/node_modules", RAW_DATA)
#: Directory names skipped at any depth, and file suffixes never packaged.
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset({"__pycache__", ".pytest_cache", ".venv", "node_modules"})
EXCLUDED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")
ZIP_CONTAINER_SUFFIXES: tuple[str, ...] = (".pptx", ".docx", ".xlsx", ".zip")
MAX_HITS_PER_PATTERN = 3
#: Separator between path components: ``\`` or ``/``, also JSON- or doubly-escaped (``\\``, ``\\\\``).
_SEPARATOR = rb"[\\/]{1,4}"
_GENERIC_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})


@dataclass(frozen=True)
class Hit:
    """One machine identifier found in a packaged file."""

    location: str        # project-relative path, ``archive.pptx!member.xml`` inside a container
    kind: str            # "home directory", "user name", "host name"
    context: str         # surrounding text with the identifier masked

    def __str__(self) -> str:
        return f"{self.location}: {self.kind} ... {self.context} ..."


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def _excluded(relative: Path, extra: Iterable[str]) -> bool:
    text = relative.as_posix()
    if any(text == prefix or text.startswith(prefix + "/") for prefix in extra):
        return True
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
        return True
    return relative.suffix.lower() in EXCLUDED_SUFFIXES


def collect_files(root: Path, *, include_raw: bool = False, skip: Iterable[Path] = ()) -> list[Path]:
    """Files to package (absolute paths, sorted by project-relative path).

    ``skip`` lists files never packaged even when inside an included folder (the output zip).
    """
    root = root.resolve()
    excluded = [p for p in EXCLUDED_PATHS if not (include_raw and p == RAW_DATA)]
    skipped = {Path(p).resolve() for p in skip}
    files: set[Path] = set()
    for source in list(INCLUDED) + ([RAW_DATA] if include_raw else []):
        base = root / source
        if base.is_file():
            candidates: Iterable[Path] = [base]
        elif base.is_dir():
            candidates = (p for p in base.rglob("*") if p.is_file())
        else:
            continue
        for path in candidates:
            if not _excluded(path.relative_to(root), excluded) and path.resolve() not in skipped:
                files.add(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


# ---------------------------------------------------------------------------
# Scan for machine identifiers
# ---------------------------------------------------------------------------
def leak_patterns(home: PurePath | str | None = None, hostname: str | None = None,
                  user: str | None = None) -> dict[str, re.Pattern[bytes]]:
    """Byte patterns of this machine's identifiers (computed now; arguments override for tests).

    * home directory - its components joined by any separator style (``C:\\Users\\name``,
      ``C:/Users/name``, JSON-escaped ``C:\\\\Users\\\\name``, ``/c/Users/name``), case-insensitive;
    * user name - inside a pytest temporary path (``pytest-of-<user>``), which survives even when the
      home prefix was shortened to ``~``;
    * host name - as a whole word, case-insensitive (skipped when shorter than 3 characters or
      ``localhost``).
    """
    patterns: dict[str, re.Pattern[bytes]] = {}
    home_path = (home if isinstance(home, PurePath) else Path(home)) if home is not None else _home()
    if home_path is not None:
        parts = [part for part in home_path.parts[1:] if part]
        if parts:
            body = _SEPARATOR.join(re.escape(part.encode("utf-8")) for part in parts)
            patterns["home directory"] = re.compile(_SEPARATOR + body + rb"(?![A-Za-z0-9_])", re.IGNORECASE)
    user_name = user if user is not None else _user()
    if user_name:
        patterns["user name"] = re.compile(rb"pytest-of-" + re.escape(user_name.encode("utf-8")) + rb"(?![A-Za-z0-9_])",
                                           re.IGNORECASE)
    host = hostname if hostname is not None else _hostname()
    if host and len(host) >= 3 and host.lower() not in _GENERIC_HOSTNAMES:
        patterns["host name"] = re.compile(rb"(?<![A-Za-z0-9-])" + re.escape(host.encode("utf-8")) + rb"(?![A-Za-z0-9-])",
                                           re.IGNORECASE)
    return patterns


def _home() -> Path | None:
    try:
        return Path.home()
    except (RuntimeError, OSError, KeyError):
        return None


def _user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no user name available: skip that pattern
        home = _home()
        return home.name if home is not None else None


def _hostname() -> str | None:
    try:
        return socket.gethostname()
    except OSError:
        return None


def _mask(context: bytes, patterns: dict[str, re.Pattern[bytes]]) -> str:
    for kind, pattern in patterns.items():
        context = pattern.sub(b"<" + kind.replace(" ", "-").encode() + b">", context)
    text = context.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", "".join(ch if ch.isprintable() else "." for ch in text)).strip()


def scan_bytes(data: bytes, location: str, patterns: dict[str, re.Pattern[bytes]]) -> list[Hit]:
    """Identifiers in ``data`` (binary data is also searched as UTF-16 text)."""
    blobs = [data]
    if b"\x00" in data:
        blobs += [data[offset:].decode("utf-16-le", errors="ignore").encode("utf-8", errors="ignore")
                  for offset in (0, 1)]
    hits: list[Hit] = []
    seen: set[tuple[str, str]] = set()
    for blob in blobs:
        for kind, pattern in patterns.items():
            for count, match in enumerate(pattern.finditer(blob)):
                if count >= MAX_HITS_PER_PATTERN:
                    break
                context = _mask(blob[max(0, match.start() - 40):match.end() + 40], patterns)
                if (kind, context) not in seen:
                    seen.add((kind, context))
                    hits.append(Hit(location, kind, context))
    return hits


def scan_file(path: Path, location: str, patterns: dict[str, re.Pattern[bytes]]) -> list[Hit]:
    """Identifiers in one file; zip containers (.pptx/.docx/.xlsx/.zip) are also scanned member by member."""
    if not patterns:
        return []
    data = path.read_bytes()
    hits = scan_bytes(data, location, patterns)
    if path.suffix.lower() in ZIP_CONTAINER_SUFFIXES and zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir():
                        hits += scan_bytes(archive.read(member), f"{location}!{member.filename}", patterns)
        except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
            hits.append(Hit(location, "unreadable container", f"{type(exc).__name__}: {exc}"))
    return hits


def scan_files(files: Sequence[Path], root: Path, patterns: dict[str, re.Pattern[bytes]]) -> list[Hit]:
    hits: list[Hit] = []
    for path in files:
        hits += scan_file(path, path.relative_to(root).as_posix(), patterns)
    return hits


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n_bytes} B"  # pragma: no cover


def write_zip(files: Sequence[Path], root: Path, out: Path, folder: str = ARCHIVE_FOLDER) -> Path:
    """Write ``files`` under ``folder/`` into ``out`` (via a temporary file, so no partial zip is left)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, f"{folder}/{path.relative_to(root).as_posix()}")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the BedSynth AI submission zip (deliverables only, "
                                                 "checked for home-directory paths and the host name).")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"zip to write (default: {DEFAULT_OUT.as_posix()} in the project root)")
    parser.add_argument("--include-raw", action="store_true", help="also package data/raw/ (the simulated source extract)")
    parser.add_argument("--dry-run", action="store_true", help="list the files, their total size and the scan result; "
                                                               "write nothing")
    parser.add_argument("--allow-leaks", action="store_true",
                        help="write the zip even if the scan finds the home directory or the host name")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    out = args.out if args.out is not None else root / DEFAULT_OUT
    out = (out if out.is_absolute() else Path.cwd() / out).resolve()
    files = collect_files(root, include_raw=args.include_raw, skip=[out])
    if not files:
        print(f"Nothing to package under {root.name}/.", file=sys.stderr)
        return 2
    total = sum(path.stat().st_size for path in files)
    if args.dry_run:
        for path in files:
            print(f"{human_size(path.stat().st_size):>10}  {path.relative_to(root).as_posix()}")
    print(f"{len(files)} files, {human_size(total)} before compression"
          + (" (data/raw included)" if args.include_raw else ""))

    patterns = leak_patterns()
    hits = scan_files(files, root, patterns)
    if hits:
        print(f"\nMachine-identifying data found ({len(hits)} hit(s) in {len({h.location.split('!')[0] for h in hits})} "
              f"file(s)):", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
    else:
        print(f"Scan: no home-directory path, pytest user path or host name in the {len(files)} files "
              f"(checked: {', '.join(patterns) or 'nothing - no identifiers could be determined'}).")
    if args.dry_run:
        print("Dry run: nothing written.")
        return 1 if hits and not args.allow_leaks else 0
    if hits and not args.allow_leaks:
        print("Aborted: nothing written. Remove the identifiers (or pass --allow-leaks to package anyway).",
              file=sys.stderr)
        return 1
    write_zip(files, root, out)
    try:
        shown = out.relative_to(root).as_posix()
    except ValueError:
        shown = out.name
    print(f"Wrote {shown} ({human_size(out.stat().st_size)}, {len(files)} files"
          + (", identifiers present: --allow-leaks" if hits else "") + ").")
    return 0


if __name__ == "__main__":
    sys.exit(main())
