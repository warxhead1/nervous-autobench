"""CodeForces scraper for LCB-style problem ingestion.

Fetches problems from the CodeForces API, respects rate limits,
and writes Problem records to data/codeforces_problems.json.

Schema: schemas/codeforces_problem.v1.json

Usage:
    scraper = CodeForcesScraper()
    problems = scraper.fetch_all()        # full ingest (~3000 problems)
    problem = scraper.fetch_one("1234A")  # single problem by CF ID
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CF_API_PROBLEMSET = "https://codeforces.com/api/problemset.problems"
CF_API_STATUS = "https://codeforces.com/api/problemset.status"
CACHE_FILE = Path(__file__).parent / "data" / "codeforces_problems.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds, exponential backoff


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------


class CodeForcesScraper:
    """Scraper for CodeForces problemset.

    Fetches from `https://codeforces.com/api/problemset.problems` — returns
    all ~3000 problems in one call. Deduplicates by cf_id and maps rating
    to difficulty tier (easy/medium/hard).

    Rate limit handling: exponential backoff on 429. Unauthenticated requests
    should stay ~1 req / 2s to avoid IP bans.
    """

    def __init__(
        self,
        cache_file: Path | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.cache_file = cache_file or CACHE_FILE
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "autobench-codeforces-scraper/1.0",
        })

    def _request(self, url: str) -> dict[str, Any]:
        """Make a single API request with retry and backoff."""
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                r = self._session.get(url, timeout=self.timeout)
                if r.status_code == 429:
                    sleep_secs = BACKOFF_BASE * (2 ** attempt)
                    time.sleep(sleep_secs)
                    continue
                r.raise_for_status()
                data = r.json()
                if data.get("status") != "OK":
                    raise RuntimeError(f"CF API error: {data.get('comment', data.get('status'))}")
                return data
            except requests.RequestException as e:
                last_err = e
                sleep_secs = BACKOFF_BASE * (2 ** attempt)
                time.sleep(sleep_secs)
        raise RuntimeError(f"CodeForces API failed after {MAX_RETRIES} attempts: {last_err}")

    def _map_rating(self, rating: int | None) -> str:
        """Map CF rating to difficulty tier."""
        if not rating:
            return "medium"
        if rating < 1200:
            return "easy"
        if rating < 2000:
            return "medium"
        return "hard"

    def fetch_one(self, cf_id: str) -> dict[str, Any] | None:
        """Fetch a single problem by CF ID (e.g. '1234A').

        Returns None if the problem is not found or the API fails.
        """
        # cf_id format: contestId + index letter e.g. "1234A"
        try:
            contest_id = int(cf_id[:-1])
            index = cf_id[-1]
        except ValueError:
            return None

        url = f"{CF_API_STATUS}?contestId={contest_id}&index={index}"
        try:
            data = self._request(url)
            # Status API returns submissions; we just need problem metadata
            # Fall back to problemset API for problem details
            return self._fetch_problem_details(contest_id, index)
        except Exception:
            return None

    def _fetch_problem_details(self, contest_id: int, index: str) -> dict[str, Any] | None:
        """Fetch single problem details from problemset API."""
        data = self._request(CF_API_PROBLEMSET)
        for p in data.get("result", {}).get("problems", []):
            if p.get("contestId") == contest_id and p.get("index") == index:
                return self._normalize_problem(p)
        return None

    def fetch_all(self) -> list[dict[str, Any]]:
        """Fetch all problems from CodeForces problemset API.

        Returns a list of normalized problem records deduplicated by cf_id.
        Writes results to cache file on success.
        """
        data = self._request(CF_API_PROBLEMSET)
        seen: set[str] = set()
        problems: list[dict[str, Any]] = []

        for p in data.get("result", {}).get("problems", []):
            cf_id = f"{p['contestId']}{p['index']}"
            if cf_id in seen:
                continue
            seen.add(cf_id)
            problems.append(self._normalize_problem(p))

        # Sort by contest_id then index for consistent ordering
        problems.sort(key=lambda x: (x["contest_id"], x["index"]))

        # Write to cache
        self._write_cache(problems)
        return problems

    def _normalize_problem(self, p: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw CF API problem into a CodeforcesProblem record."""
        cf_id = f"{p['contestId']}{p['index']}"
        rating = p.get("rating")
        return {
            "cf_id": cf_id,
            "contest_id": p["contestId"],
            "index": p["index"],
            "title": p.get("name", ""),
            "statement": "",  # statement requires separate HTML fetch — skip for v1
            "input_format": p.get("inputFormat", ""),
            "output_format": p.get("outputFormat", ""),
            "difficulty": self._map_rating(rating),
            "tags": p.get("tags", []),
            "sample_tests": [],  # sample tests require separate fetch — skip for v1
            "metadata": {
                "cf_url": f"https://codeforces.com/contest/{p['contestId']}/problem/{p['index']}",
                "rating": rating,
                "solved_by": p.get("solvedCount"),
                "scraped_at": "",  # filled in by _write_cache
            },
        }

    def _write_cache(self, problems: list[dict[str, Any]]) -> None:
        """Write problems list to cache file with timestamp."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cache = {
            "scraped_at": now,
            "count": len(problems),
            "problems": problems,
        }
        with open(self.cache_file, "w") as fh:
            json.dump(cache, fh, indent=2)

    def load_cached(self) -> list[dict[str, Any]]:
        """Load problems from cache file. Returns empty list if cache missing."""
        if not self.cache_file.exists():
            return []
        try:
            with open(self.cache_file) as fh:
                cache = json.load(fh)
            return cache.get("problems", [])
        except (json.JSONDecodeError, OSError):
            return []

    def to_benchmark_cases(
        self,
        problems: list[dict[str, Any]] | None = None,
        difficulty_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert CodeforcesProblem records to BenchmarkCase dicts.

        Args:
            problems: List of problem records. If None, loads from cache.
            difficulty_filter: If set, only include problems of this difficulty.

        Returns:
            List of BenchmarkCase-compatible dicts.
        """
        if problems is None:
            problems = self.load_cached()

        cases = []
        for p in problems:
            if difficulty_filter and p.get("difficulty") != difficulty_filter:
                continue
            cases.append({
                "id": p["cf_id"],
                "prompt": f"{p['title']}\n\n{p.get('statement', '')}\n\nInput: {p.get('input_format', 'See problem URL')}\nOutput: {p.get('output_format', 'See problem URL')}",
                "language": "python",
                "expected_output": "",  # use sample_tests when available
                "constraints": {
                    "max_time_seconds": 10,
                    "max_memory_mb": 512,
                },
                "metadata": {
                    "source": "codeforces",
                    "difficulty": p.get("difficulty"),
                    "tags": p.get("tags", []),
                    "rating": p.get("metadata", {}).get("rating"),
                    "cf_url": p.get("metadata", {}).get("cf_url"),
                },
            })
        return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CodeForces problem scraper")
    parser.add_argument("--fetch-all", action="store_true", help="Fetch all problems")
    parser.add_argument("--fetch-one", metavar="CF_ID", help="Fetch single problem (e.g. 1234A)")
    parser.add_argument("--list", action="store_true", help="List cached problems")
    parser.add_argument("--diff", metavar="DIFF", choices=["easy", "medium", "hard"], help="Filter by difficulty")
    args = parser.parse_args()

    scraper = CodeForcesScraper()

    if args.fetch_all:
        print(f"Fetching all Codeforces problems (rate-limit aware)...")
        problems = scraper.fetch_all()
        print(f"  → {len(problems)} problems written to {scraper.cache_file}")
        diff_counts = {"easy": 0, "medium": 0, "hard": 0}
        for p in problems:
            diff_counts[p["difficulty"]] += 1
        for d, c in diff_counts.items():
            print(f"    {d}: {c}")

    elif args.fetch_one:
        print(f"Fetching problem {args.fetch_one}...")
        p = scraper.fetch_one(args.fetch_one)
        if p:
            print(f"  → {p['title']} [{p['difficulty']}]")
        else:
            print("  → not found")

    elif args.list:
        problems = scraper.load_cached()
        print(f"Cached: {len(problems)} problems")
        for p in problems[:5]:
            print(f"  {p['cf_id']}: {p['title']} [{p['difficulty']}]")
        if len(problems) > 5:
            print(f"  ... and {len(problems) - 5} more")

    else:
        parser.print_help()