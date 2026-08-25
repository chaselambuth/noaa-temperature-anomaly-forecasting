"""One-command pre-push checks for the repository.

This script is intentionally lightweight: it validates repository hygiene and
unit tests without rerunning expensive notebooks or model experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".dockerignore",
    ".gitignore",
    ".html",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yml",
    ".yaml",
}

GENERATED_OR_LOCAL_DIRS = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "catboost_info",
    "runs",
}

LOCAL_PATH_NEEDLES = (
    "C:/Users/chase",
    r"C:\Users\chase",
    "Users/chase",
    "NOAA Personal Project",
    "Project root:",
    "src path added:",
)

DATA_DIR_NAMES = {"noaa_raw_data", "sst_cache"}
DATA_SUFFIXES = {".nc"}
DATA_FILENAMES = {"nina34.anom.csv"}


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def iter_files() -> Iterable[Path]:
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in GENERATED_OR_LOCAL_DIRS for part in path.parts):
            continue
        yield path


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore", ".dockerignore"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def notebook_paths(skip_notebooks: set[str]) -> list[Path]:
    notebooks = sorted((REPO_ROOT / "experiment_notebooks").glob("*.ipynb"))
    return [path for path in notebooks if path.name not in skip_notebooks]


def check_python_version() -> CheckResult:
    ok = sys.version_info >= (3, 11)
    detail = f"Python {sys.version.split()[0]}"
    return CheckResult("Python version", ok, [detail])


def check_notebooks_parse(skip_notebooks: set[str]) -> CheckResult:
    details: list[str] = []
    ok = True
    for path in notebook_paths(skip_notebooks):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            details.append(f"{rel(path)} parses")
        except Exception as exc:  # noqa: BLE001 - report any notebook parse failure.
            ok = False
            details.append(f"{rel(path)} failed to parse: {exc}")
    return CheckResult("Notebook JSON validity", ok, details)


def check_notebook_error_outputs(skip_notebooks: set[str]) -> CheckResult:
    details: list[str] = []
    ok = True
    for path in notebook_paths(skip_notebooks):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(nb.get("cells", [])):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    ok = False
                    ename = output.get("ename", "error")
                    evalue = output.get("evalue", "")
                    details.append(f"{rel(path)} cell {idx}: {ename}: {evalue}")
    if not details:
        details.append("No notebook error outputs found")
    return CheckResult("Notebook error outputs", ok, details)


def check_text_leakage(skip_notebooks: set[str]) -> CheckResult:
    details: list[str] = []
    ok = True
    for path in iter_files():
        if path.name in skip_notebooks:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if not is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in LOCAL_PATH_NEEDLES:
            if needle in text:
                ok = False
                details.append(f"{rel(path)} contains {needle!r}")
                break
    if not details:
        details.append("No local machine paths found")
    return CheckResult("Portable text scan", ok, details)


def check_data_files(allow_local_data: bool) -> CheckResult:
    details: list[str] = []
    offenders: list[Path] = []
    for path in iter_files():
        lower_parts = {part.lower() for part in path.parts}
        if lower_parts & DATA_DIR_NAMES:
            offenders.append(path)
            continue
        if path.suffix.lower() in DATA_SUFFIXES:
            offenders.append(path)
            continue
        if path.name.lower() in DATA_FILENAMES:
            offenders.append(path)

    if offenders:
        shown = offenders[:20]
        details.extend(f"{rel(path)}" for path in shown)
        if len(offenders) > len(shown):
            details.append(f"... and {len(offenders) - len(shown)} more")
    else:
        details.append("No local data files found")

    if allow_local_data and offenders:
        details.insert(0, "Local data files present but allowed by flag")
        return CheckResult("Local data files", True, details)
    return CheckResult("Local data files", not offenders, details)


def check_gitignore() -> CheckResult:
    gitignore = REPO_ROOT / ".gitignore"
    required = ["noaa_raw_data/", "*.nc", "nina34.anom.csv", "sst_cache/"]
    if not gitignore.exists():
        return CheckResult(".gitignore data exclusions", False, [".gitignore is missing"])

    text = gitignore.read_text(encoding="utf-8")
    missing = [pattern for pattern in required if pattern not in text]
    if missing:
        return CheckResult(".gitignore data exclusions", False, [f"Missing: {', '.join(missing)}"])
    return CheckResult(".gitignore data exclusions", True, ["Required data exclusions are present"])


def check_tests(run_tests: bool) -> CheckResult:
    if not run_tests:
        return CheckResult("Unit tests", True, ["Skipped by flag"])

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else os.pathsep.join([src_path, existing])

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout.strip().splitlines()
    tail = output[-12:] if output else ["pytest produced no output"]
    return CheckResult("Unit tests", proc.returncode == 0, tail)


def print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {result.name}")
    for line in result.details:
        print(f"  {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-push checks for the repository.")
    parser.add_argument(
        "--skip-notebook",
        action="append",
        default=[],
        metavar="NAME",
        help="Skip a notebook by filename, useful while a notebook is actively running.",
    )
    parser.add_argument(
        "--allow-local-data",
        action="store_true",
        help="Warn-but-pass when ignored local NOAA data files are present.",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip pytest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    skip_notebooks = set(args.skip_notebook)

    checks = [
        check_python_version(),
        check_gitignore(),
        check_notebooks_parse(skip_notebooks),
        check_notebook_error_outputs(skip_notebooks),
        check_text_leakage(skip_notebooks),
        check_data_files(args.allow_local_data),
        check_tests(run_tests=not args.no_tests),
    ]

    print(f"Pre-push check root: {REPO_ROOT}")
    if skip_notebooks:
        print(f"Skipped notebooks: {', '.join(sorted(skip_notebooks))}")
    print()

    for result in checks:
        print_result(result)
        print()

    failed = [result.name for result in checks if not result.ok]
    if failed:
        print("Pre-push checks failed:")
        for name in failed:
            print(f"  - {name}")
        return 1

    print("All pre-push checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
