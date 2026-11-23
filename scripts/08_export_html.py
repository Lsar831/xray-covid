"""Export the Markdown report to a self-contained HTML file.

Markdown has no alignment syntax, so justified text is applied through
reports/report.css at export time. Run this after scripts/05_build_report.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SOURCE = REPORTS / "ANALYSIS_REPORT.md"
STYLESHEET = REPORTS / "report.css"
TARGET = REPORTS / "ANALYSIS_REPORT.html"


def main() -> int:
    if not SOURCE.exists():
        print(f"Report not found: {SOURCE}. Run scripts/05_build_report.py first.", file=sys.stderr)
        return 1
    if not STYLESHEET.exists():
        print(f"Stylesheet not found: {STYLESHEET}", file=sys.stderr)
        return 1

    command = [
        "pandoc",
        str(SOURCE),
        "--from", "markdown",
        "--to", "html5",
        "--standalone",
        "--embed-resources",
        "--css", STYLESHEET.name,
        "--resource-path", str(REPORTS),
        "--metadata", "title=COVID-19 and Other Pneumonia Etiology Classification in Chest X-ray",
        "--output", str(TARGET),
    ]
    try:
        subprocess.run(command, check=True, cwd=REPORTS)
    except FileNotFoundError:
        print("pandoc is not installed. Install it and run this script again.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"pandoc failed with exit code {error.returncode}", file=sys.stderr)
        return error.returncode

    print(f"HTML report written to {TARGET} ({TARGET.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
