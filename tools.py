from __future__ import annotations

import json
import os
import tqdm
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote
import requests
import ruamel.yaml as yaml
import pandas as pd


def normalize_doi(raw: str):
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None

    s = s.lower()

    # Common prefixes / URL forms
    for prefix in (
        "doi:",
        "doi.org/",
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()

    # Sometimes people paste a full URL with query params/fragments; strip conservatively
    # (DOIs themselves can contain many characters, but '?' and '#' are typically URL parts)
    s = s.split("?", 1)[0].split("#", 1)[0].strip()

    # Very light validation
    if not s.startswith("10."):
        return None

    return s

    # De-duplicate by normalized DOI but keep first original string
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for original, norm in items:
        if norm not in seen:
            seen.add(norm)
            deduped.append((original, norm))
    return deduped


def openalex_lookup(
    session,
    normalized_doi,
    *,
    timeout: int = 30,
    max_retries: int = 6,
    base_sleep: float = 0.6,
    polite_jitter: float = 0.2,
) -> Dict[str, object]:
    """
    Returns a dict with keys:
      status: "ok" | "not_found" | "error"
      cited_by_count: int | None
      openalex_id: str | None
      error: str | None
    """
    # OpenAlex DOI endpoint format:
    url = f"https://api.openalex.org/works/https://doi.org/{quote(normalized_doi)}"

    headers = {}

    for attempt in range(max_retries + 1):
        try:
            print("Get", url)
            resp = session.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                return {
                    "status": "not_found",
                    "cited_by_count": None,
                    "openalex_id": None,
                    "error": None,
                }

            # Handle rate-limiting / transient server errors with retry
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)

            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "ok",
                "cited_by_count": data.get("cited_by_count"),
                "publication_year": data.get("publication_year"),
                "openalex_id": data.get("id"),
                "error": None,
            }

        except Exception as e:
            # If this was the last attempt, fail
            if attempt >= max_retries:
                return {
                    "status": "error",
                    "cited_by_count": None,
                    "openalex_id": None,
                    "error": str(e),
                }

            # Exponential backoff with jitter
            sleep_s = (base_sleep * (2**attempt)) + random.uniform(0, polite_jitter)
            time.sleep(sleep_s)

    # Should never reach here
    return {
        "status": "error",
        "cited_by_count": None,
        "openalex_id": None,
        "error": "unexpected",
    }


def get_openalex_results(dois):

    cachefile = "openalex_cache.json"
    cache = {}
    if os.path.exists(cachefile):
        with open(cachefile, "r") as f:
            cache = json.load(f)
    results = {}
    with requests.Session() as session:
        for doi in tqdm.tqdm(dois):
            if doi in cache:
                results[doi] = cache[doi]
            else:
                r = openalex_lookup(session, doi)
                results[doi] = {
                    "citations": r.get("cited_by_count"),
                    "year": r.get("publication_year"),
                    "openalex_id": r.get("openalex_id"),
                }
                # Polite pacing on success
                time.sleep(0.1 + random.uniform(0, 0.05))

    with open(cachefile, "w") as f:
        json.dump(results, f)

    return results


def main():
    with open("tools.yaml") as f:
        d = yaml.YAML(typ="rt").load(f)

    df = pd.DataFrame(d)
    print("Total = ", df.shape[0])
    df = df[~df["publication"].isnull()]
    print(df[["name", "type", "interface", "package"]].sort_values("type"))

    print(df["publication"])
    df["doi"] = [normalize_doi(doi) for doi in df["publication"]]
    oa_results = get_openalex_results(df["doi"])
    df["citations"] = [oa_results[doi]["citations"] for doi in df["doi"]]
    df["year"] = [oa_results[doi]["year"] for doi in df["doi"]]
    # print(oa_results)
    # print(df)

    sections = ["simulation", "arg_inference", "parameter_inference",
            "statgen", "visualisation", "format_conversion", "analysis"]

    for section in sections:
        dfs = df[df["type"] == section][["name", "interface", "package", "year", "citations"]]
        dfs = dfs.sort_values(["year", "citations"], ascending=False)
        print(dfs.to_latex(index=False, na_rep='', float_format="%.0f"))


if __name__ == "__main__":
    main()
