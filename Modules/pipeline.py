"""
pipeline.py — CLI entry point for the AI Research Trend Analyzer.

Usage:
  python pipeline.py --fetch                       # Fetch papers from arXiv
  python pipeline.py --analyze                     # Topic model + trend analysis
  python pipeline.py --dashboard                   # Generate HTML dashboard
  python pipeline.py --all                         # Full pipeline
  python pipeline.py --all --test                  # Quick test (~80 papers)
  python pipeline.py --fetch --max-papers 2000
  python pipeline.py --fresh                       # Clear all cached data and re-run
"""

import argparse
import os
import sys
import shutil

from config import (
    DATA_DIR, DASHBOARD_FILE, DRIFT_FILE, PAPERS_FILE,
    PROGRESS_FILE, TOPIC_FILE, TREND_FILE,
)

BANNER = """
╔════════════════════════════════════════════════════════════╗
║   Detecting Emerging Research Trends in AI Literature      ║
║   Multi-Keyword Temporal Analysis · Topic Drift Detection  ║
╚════════════════════════════════════════════════════════════╝
"""


def cmd_fetch(args):
    from scraper import fetch_papers, save_papers
    papers = fetch_papers(
        max_papers=args.max_papers,
        test_mode=args.test,
        year_start=args.year_start,
    )
    save_papers(papers)


def cmd_analyze(args):
    if not os.path.exists(PAPERS_FILE):
        print(f"✗  {PAPERS_FILE} not found. Run --fetch first.")
        sys.exit(1)
    from analyzer import run_analysis
    run_analysis()


def cmd_dashboard(args):
    for path in [TREND_FILE, DRIFT_FILE]:
        if not os.path.exists(path):
            print(f"✗  {path} not found. Run --analyze first.")
            sys.exit(1)
    from dashboard import run_dashboard
    run_dashboard()


def cmd_fresh(args):
    """Wipe all cached data files."""
    targets = [DATA_DIR, DASHBOARD_FILE]
    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t)
            print(f" Removed directory: {t}/")
        elif os.path.exists(t):
            os.remove(t)
            print(f" Removed: {t}")
    print(" Fresh start ready.\n")


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="AI Research Trend Analyzer — arXiv Temporal Analysis with Topic Drift Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Stage flags
    parser.add_argument("--fetch",     action="store_true", help="Fetch papers from arXiv")
    parser.add_argument("--analyze",   action="store_true", help="Run topic modeling + trend analysis")
    parser.add_argument("--dashboard", action="store_true", help="Generate HTML dashboard")
    parser.add_argument("--all",       action="store_true", help="Run full pipeline (fetch → analyze → dashboard)")
    parser.add_argument("--fresh",     action="store_true", help="Clear all cached data before running")

    # Options
    parser.add_argument("--max-papers", type=int, default=3000,
                        help="Max papers to fetch (default: 3000)")
    parser.add_argument("--year-start", type=int, default=2019,
                        help="Earliest publication year to include (default: 2019)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: fetch ~80 papers only")

    args = parser.parse_args()

    # Require at least one action
    if not any([args.fetch, args.analyze, args.dashboard, args.all, args.fresh]):
        parser.print_help()
        sys.exit(0)

    print(BANNER)
    print(f"  Mode        : {'TEST' if args.test else 'FULL'}")
    if args.fetch or args.all:
        print(f"  Max papers  : {'~80' if args.test else args.max_papers}")
        print(f"  Year start  : {args.year_start}")
    print()

    # --fresh wipes state before anything else
    if args.fresh:
        cmd_fresh(args)

    run_all = args.all

    if args.fetch or run_all:
        print("\n STEP 1: Fetching Papers ──────────────────────────────────")
        cmd_fetch(args)

    if args.analyze or run_all:
        print("\n STEP 2: Analyzing Trends ─────────────────────────────────")
        cmd_analyze(args)

    if args.dashboard or run_all:
        print("\n STEP 3: Generating Dashboard ─────────────────────────────")
        cmd_dashboard(args)

    print("\n All done!")
    if args.dashboard or run_all:
        print(f"  Open ./{DASHBOARD_FILE} in your browser.\n")


if __name__ == "__main__":
    main()
