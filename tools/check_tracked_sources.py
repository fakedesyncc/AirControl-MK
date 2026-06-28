"""Fail when source files exist locally but are not tracked by git.

This catches release mistakes where tests pass in the developer worktree because
an ignored file exists locally, but GitHub Actions fails after checkout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("aircontrol", "tests", "tools", "packaging")


def _git_lines(args: list[str]) -> set[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _is_git_checkout() -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def main() -> int:
    if not _is_git_checkout():
        print("Tracked source check skipped: not a git checkout.")
        return 0

    tracked = _git_lines(["ls-files"])
    missing: list[str] = []

    for source_root in SOURCE_ROOTS:
        root = ROOT / source_root
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel not in tracked:
                missing.append(rel)

    if missing:
        print("Source files exist locally but are not tracked by git:", file=sys.stderr)
        for rel in missing:
            print(f"- {rel}", file=sys.stderr)
        return 1

    print("Tracked source check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
