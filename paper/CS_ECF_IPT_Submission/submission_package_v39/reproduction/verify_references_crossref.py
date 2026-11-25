"""Verify BibTeX records against Crossref metadata."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def _balanced_value(text: str, start: int) -> str:
    opener = text[start]
    closer = "}" if opener == "{" else '"'
    if opener == '"':
        end = text.find('"', start + 1)
        if end < 0:
            raise ValueError("unterminated quoted BibTeX value")
        return text[start + 1 : end]

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError("unterminated braced BibTeX value")


def parse_entries(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries = []
    position = 0
    header = re.compile(r"@(\w+)\s*\{\s*([^,]+),")
    while match := header.search(text, position):
        key = match.group(2).strip()
        body_start = match.end()
        depth = 1
        index = body_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        body = text[body_start : index - 1]
        fields = {"ENTRYTYPE": match.group(1).lower(), "ID": key}
        field_pattern = re.compile(r"(?m)^\s*([A-Za-z][\w-]*)\s*=\s*")
        for field_match in field_pattern.finditer(body):
            value_start = field_match.end()
            while value_start < len(body) and body[value_start].isspace():
                value_start += 1
            if value_start >= len(body) or body[value_start] not in '{"':
                continue
            fields[field_match.group(1).lower()] = _balanced_value(
                body, value_start
            )
        entries.append(fields)
        position = index
    return entries


def normalize(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = re.sub(r"[{}\\]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def normalize_range(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(r"[-–]+", "-", text)


def crossref_year(message: dict) -> int | None:
    for key in ("published-print", "published-online", "issued"):
        parts = message.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def verify(entry: dict[str, str]) -> dict:
    doi = entry.get("doi")
    result = {"key": entry["ID"], "doi": doi, "issues": []}
    if not doi:
        result["status"] = "no-doi"
        return result

    url = "https://api.crossref.org/works/" + requests.utils.quote(doi)
    response = None
    with requests.Session() as session:
        for attempt in range(4):
            try:
                response = session.get(
                    url,
                    headers={
                        "User-Agent": (
                            "CS-ECF-reference-audit/1.1 "
                            "(mailto:zgwu@ncepu.edu.cn)"
                        )
                    },
                    timeout=40,
                )
                if response.status_code in (200, 404):
                    break
            except requests.RequestException:
                pass
            time.sleep(1 + attempt)

    if response is None or response.status_code != 200:
        result["status"] = "error"
        result["http_status"] = None if response is None else response.status_code
        result["issues"].append("DOI did not resolve through Crossref")
        return result

    message = response.json()["message"]
    result["status"] = "verified"
    bib_title = normalize(entry.get("title", ""))
    crossref_title = normalize((message.get("title") or [""])[0])
    result["title_similarity"] = round(
        difflib.SequenceMatcher(None, bib_title, crossref_title).ratio(), 4
    )
    if result["title_similarity"] < 0.86:
        result["issues"].append("DOI title differs from the BibTeX title")

    bib_year = re.search(r"\d{4}", entry.get("year", ""))
    remote_year = crossref_year(message)
    result["year"] = [bib_year.group(0) if bib_year else None, remote_year]
    if bib_year and remote_year and int(bib_year.group(0)) != remote_year:
        result["issues"].append("publication year differs")

    bib_volume = entry.get("volume")
    remote_volume = message.get("volume")
    result["volume"] = [bib_volume, remote_volume]
    if (
        bib_volume
        and remote_volume
        and normalize_range(bib_volume) != normalize_range(remote_volume)
    ):
        result["issues"].append("volume differs")

    bib_pages = entry.get("pages")
    remote_pages = message.get("page")
    result["pages"] = [bib_pages, remote_pages]
    if bib_pages and remote_pages:
        first_bib = re.split(r"[-–]", bib_pages)[0]
        first_remote = re.split(r"[-–]", remote_pages)[0]
        if normalize_range(first_bib) != normalize_range(first_remote):
            result["issues"].append("first page differs")

    bib_authors = [
        author.strip()
        for author in re.split(r"\s+and\s+", entry.get("author", ""))
        if author.strip()
    ]
    remote_authors = message.get("author", [])
    result["author_count"] = [len(bib_authors), len(remote_authors)]
    if bib_authors and remote_authors and len(bib_authors) != len(remote_authors):
        result["issues"].append("author count differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    entries = parse_entries(args.bib)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(verify, entries))

    issue_count = sum(len(item["issues"]) for item in results)
    summary = {
        "entries": len(entries),
        "doi_entries": sum(bool(item.get("doi")) for item in entries),
        "verified": sum(item["status"] == "verified" for item in results),
        "issue_count": issue_count,
        "results": results,
    }
    rendered = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.fail_on_issues and issue_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
