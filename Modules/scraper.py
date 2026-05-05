"""
scraper.py
Metadata-only arXiv fetcher for AI literature.
Full mode fetches papers year-by-year (2019→now) to ensure temporal spread
across quarters — essential for meaningful trend analysis.
"""

import csv
import datetime
import json
import os
import time
import xml.etree.ElementTree as ET
from threading import Lock

import requests
from tqdm import tqdm

from config import (
    AI_KEYWORDS, DATA_DIR, KEYWORDS_PER_BATCH,
    PAPERS_FILE, PROGRESS_FILE, RESULTS_PER_BATCH, YEAR_START,
)

ARXIV_API = "https://arxiv.org/api/query"
NS        = "http://www.w3.org/2005/Atom"


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------
class FetchProgress:
    def __init__(self, path=PROGRESS_FILE):
        self._path = path
        self._lock = Lock()
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)
        else:
            self._data = {"papers": {}, "kw_map": {}}

    def has(self, aid: str) -> bool:
        return aid in self._data["papers"]

    def add(self, aid: str, record: dict, keywords: list):
        with self._lock:
            self._data["papers"][aid] = record
            existing = set(self._data["kw_map"].get(aid, []))
            self._data["kw_map"][aid] = sorted(existing | set(keywords))
            self._save()

    def all_papers(self) -> list:
        result = []
        for aid, rec in self._data["papers"].items():
            rec = dict(rec)
            rec["matched_keywords"] = "; ".join(self._data["kw_map"].get(aid, []))
            result.append(rec)
        return result

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f)

    @property
    def count(self) -> int:
        return len(self._data["papers"])


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------
def _build_batches(keywords: list, batch_size: int = KEYWORDS_PER_BATCH) -> list:
    seen, unique = set(), []
    for kw in keywords:
        k = kw.strip().lower()
        if k and k not in seen:
            seen.add(k)
            unique.append(kw.strip())
    batches = []
    for i in range(0, len(unique), batch_size):
        chunk = unique[i:i + batch_size]
        parts = [f'ti:"{kw}" OR abs:"{kw}"' for kw in chunk]
        batches.append((" OR ".join(parts), chunk))
    return batches


# ---------------------------------------------------------------------------
# Raw HTTP fetch (accepts any fully-formed query string)
# ---------------------------------------------------------------------------
def _fetch_batch_raw(search_query: str, max_results: int) -> list:
    """POST a query directly to arXiv API with exponential backoff."""
    params = {
        "search_query": search_query,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
        "start":        0,
        "max_results":  max_results,
    }
    for attempt in range(1, 5):
        try:
            resp = requests.get(ARXIV_API, params=params,
                                timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                root    = ET.fromstring(resp.text)
                entries = []
                for entry in root.findall(f"{{{NS}}}entry"):
                    aid_url   = entry.findtext(f"{{{NS}}}id", "")
                    aid       = aid_url.split("/abs/")[-1].strip()
                    title     = (entry.findtext(f"{{{NS}}}title") or "").replace("\n", " ").strip()
                    abstract  = (entry.findtext(f"{{{NS}}}summary") or "").replace("\n", " ").strip()[:800]
                    published = (entry.findtext(f"{{{NS}}}published") or "")[:10]
                    try:
                        year  = int(published[:4])
                        month = int(published[5:7])
                    except Exception:
                        year, month = 0, 0
                    authors = "; ".join(
                        (a.findtext(f"{{{NS}}}name") or "")
                        for a in entry.findall(f"{{{NS}}}author")[:5]
                    )
                    cats = " ".join(
                        t.get("term", "")
                        for t in entry.findall(
                            "{http://arxiv.org/schemas/atom}primary_category"
                        )
                    ) or ""
                    entries.append({
                        "arxiv_id": aid, "title": title, "authors": authors,
                        "abstract": abstract, "published": published,
                        "year": year, "month": month, "categories": cats,
                    })
                return entries
            else:
                wait = 20 * attempt
                tqdm.write(f"    HTTP {resp.status_code} (attempt {attempt}/4) — waiting {wait}s …")
                time.sleep(wait)
        except Exception as exc:
            wait = 20 * attempt
            tqdm.write(f"    Error (attempt {attempt}/4): {exc} — waiting {wait}s …")
            time.sleep(wait)
    tqdm.write("    All retries failed — skipping batch.")
    return []


def _fetch_batch(query: str, year_start: int, max_results: int) -> list:
    """Fetch without date filter; filter by year in Python."""
    entries = _fetch_batch_raw(query, max_results)
    return [e for e in entries if e.get("year", 0) >= year_start]


# ---------------------------------------------------------------------------
# Main fetch entry point
# ---------------------------------------------------------------------------
def fetch_papers(max_papers: int = 3000, test_mode: bool = False,
                 year_start: int = YEAR_START) -> list:
    progress = FetchProgress()
    batches  = _build_batches(AI_KEYWORDS)
    cap      = 80 if test_mode else max_papers
    current_year = datetime.datetime.now().year

    print(f"\n{'='*60}")
    print(f"  ArXiv AI Literature Scraper")
    print(f"{'='*60}")
    print(f"  Keywords  : {len(AI_KEYWORDS)}  ->  {len(batches)} query batches")
    print(f"  Target    : {cap} papers")
    print(f"  Mode      : {'TEST (recent papers only)' if test_mode else 'FULL (year-stratified 2019→now)'}\n")

    if test_mode:
        # Test mode: just grab recent papers quickly 
        for idx, (query, kw_list) in enumerate(batches[:4]):
            if progress.count >= cap:
                break
            label = ", ".join(kw_list[:3])
            print(f"[{idx+1}/4] {label}  ({progress.count}/{cap})")
            for record in _fetch_batch(query, year_start, 40):
                aid = record["arxiv_id"]
                if progress.has(aid):
                    progress.add(aid, progress._data["papers"][aid], kw_list)
                else:
                    progress.add(aid, record, kw_list)
                if progress.count >= cap:
                    break
            print(f"  -> {progress.count} total")
            if progress.count < cap:
                time.sleep(5)

    else:
        # Full mode: year-stratified fetch for temporal spread 
        years    = list(range(year_start, current_year + 1))
        per_year = max(cap // len(years), 30)
        per_batch_size = min(50, per_year)

        print(f"  Years     : {years[0]} → {years[-1]}  (~{per_year} papers/year)\n")

        for year in years:
            year_start_count = progress.count
            year_cap = year_start_count + per_year

            print(f"--- Year {year} ---")
            for idx, (kw_query, kw_list) in enumerate(batches):
                if progress.count >= min(year_cap, cap):
                    break

                # Narrow to this specific year using date filter
                full_query = (
                    f"({kw_query}) AND "
                    f"submittedDate:[{year}01010000 TO {year}12312359]"
                )
                entries = _fetch_batch_raw(full_query, per_batch_size)

                new_ct = dup_ct = 0
                for record in entries:
                    if record.get("year", 0) != year:
                        continue
                    aid = record["arxiv_id"]
                    if progress.has(aid):
                        progress.add(aid, progress._data["papers"][aid], kw_list)
                        dup_ct += 1
                    else:
                        progress.add(aid, record, kw_list)
                        new_ct += 1
                    if progress.count >= min(year_cap, cap):
                        break

                if new_ct or dup_ct:
                    label = ", ".join(kw_list[:2])
                    print(f"  [{idx+1}] {label:<35} +{new_ct} new")

                if progress.count < min(year_cap, cap):
                    time.sleep(3)

            got = progress.count - year_start_count
            print(f"  ✓ {year}: {got} papers  (total: {progress.count}/{cap})\n")
            if progress.count >= cap:
                break

    papers = progress.all_papers()
    print(f"\n{'='*60}")
    print(f"  Total unique papers: {len(papers)}")
    print(f"{'='*60}\n")
    return papers


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def save_papers(papers: list):
    fields = [
        "arxiv_id", "title", "authors", "abstract",
        "published", "year", "month", "categories", "matched_keywords",
    ]
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PAPERS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            writer.writerow({k: p.get(k, "") for k in fields})
    print(f"Saved {len(papers)} papers -> {PAPERS_FILE}")
