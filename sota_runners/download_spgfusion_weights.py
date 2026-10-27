"""Download the official SPGFusion checkpoint with resumable range requests.

Google Drive throttles a single transfer from some networks.  The official file
accepts byte ranges, so this helper downloads independent segments in parallel
and assembles them only after every segment has the expected length.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


FILE_ID = "1I7-rqJOToGpkK-XXUsKwWZ88PTPvQTO3"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"
DOWNLOAD_PARAMS = {"id": FILE_ID, "export": "download", "confirm": "t"}
USER_AGENT = "Mozilla/5.0"


def probe_size(url: str, params: dict | None) -> int:
    response = requests.get(
        url,
        params=params,
        headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT},
        timeout=(30, 90),
    )
    response.raise_for_status()
    content_range = response.headers.get("Content-Range", "")
    try:
        return int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(
            f"Could not determine checkpoint size from {content_range!r}"
        ) from error


def download_segment(url: str, params: dict | None, part: Path,
                     start: int, end: int,
                     attempts: int) -> Path:
    expected = end - start + 1
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            return part
        if have > expected:
            part.unlink()
            have = 0
        headers = {
            "Range": f"bytes={start + have}-{end}",
            "User-Agent": USER_AGENT,
        }
        try:
            with requests.get(url, params=params, stream=True, headers=headers,
                              timeout=(30, 30)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
                with part.open("ab") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if part.stat().st_size == expected:
                return part
            raise RuntimeError(
                f"short segment {part.name}: {part.stat().st_size}/{expected}"
            )
        except (requests.RequestException, OSError, RuntimeError) as error:
            if attempt == attempts:
                raise RuntimeError(f"{part.name} failed after {attempts} attempts") from error
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def parallel_download(url: str, params: dict | None, output: Path,
                      workers: int = 8, attempts: int = 5,
                      force: bool = False, label: str = "download") -> None:
    output = output.resolve()
    total = probe_size(url, params)
    if output.exists() and output.stat().st_size == total and not force:
        print(f"[{label}] already complete: {output} ({total} bytes)")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output.with_name(output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers, 16))
    boundaries = [round(i * total / workers) for i in range(workers + 1)]
    segments = [
        (part_dir / f"part-{i:02d}", boundaries[i], boundaries[i + 1] - 1)
        for i in range(workers)
        if boundaries[i + 1] > boundaries[i]
    ]
    print(f"[{label}] {total} bytes, {len(segments)} segments -> {output}",
          flush=True)

    with ThreadPoolExecutor(max_workers=len(segments)) as pool:
        futures = [
            pool.submit(download_segment, url, params, part, start, end,
                        attempts)
            for part, start, end in segments
        ]
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"[{label}] segment {completed}/{len(segments)}",
                  flush=True)

    temporary = output.with_name(output.name + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as target:
        for part, _, _ in segments:
            with part.open("rb") as source:
                while True:
                    block = source.read(4 * 1024 * 1024)
                    if not block:
                        break
                    target.write(block)
                    digest.update(block)
    if temporary.stat().st_size != total:
        raise RuntimeError(
            f"assembled size {temporary.stat().st_size} != expected {total}"
        )
    os.replace(temporary, output)
    shutil.rmtree(part_dir)
    print(f"[{label}] complete: {output}")
    print(f"[{label}] sha256: {digest.hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="sota/SPGFusion/CHECKPOINT/best_model_1.pth",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    parallel_download(
        DOWNLOAD_URL,
        DOWNLOAD_PARAMS,
        Path(args.output),
        workers=args.workers,
        attempts=args.attempts,
        force=args.force,
        label="spg-download",
    )


if __name__ == "__main__":
    main()
