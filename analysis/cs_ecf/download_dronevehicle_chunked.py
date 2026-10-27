"""Download DroneVehicle test.zip with resumable parallel range requests."""

from __future__ import annotations

import concurrent.futures
import math
import os
import time
from pathlib import Path

import requests


URL = "https://huggingface.co/datasets/McCheng/DroneVehicle/resolve/main/test.zip"
TARGET_DIRECTORY = Path(r"D:\Code\_ssgf_external\dronevehicle_hf")
TARGET_PATH = TARGET_DIRECTORY / "test.zip"
PARTS_DIRECTORY = TARGET_DIRECTORY / "test_parts"
EXPECTED_BYTES = 4_430_343_899
CHUNK_BYTES = 16 * 1024 * 1024
MAX_WORKERS = 16
MAX_RETRIES = 5
HEADERS = {"User-Agent": "Codex-DroneVehicle-Downloader/1.0"}


def chunk_specs():
    chunks = []
    for index, start in enumerate(range(0, EXPECTED_BYTES, CHUNK_BYTES)):
        stop = min(start + CHUNK_BYTES - 1, EXPECTED_BYTES - 1)
        chunks.append((index, start, stop))
    return chunks


def download_chunk(spec):
    index, start, stop = spec
    expected = stop - start + 1
    part_path = PARTS_DIRECTORY / f"{index:04d}.part"
    if part_path.exists() and part_path.stat().st_size == expected:
        return index, expected, True

    temporary_path = part_path.with_suffix(".tmp")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {**HEADERS, "Range": f"bytes={start}-{stop}"}
            with requests.get(
                URL,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Expected 206, got {response.status_code}"
                    )
                written = 0
                with temporary_path.open("wb") as handle:
                    for block in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not block:
                            continue
                        handle.write(block)
                        written += len(block)
                if written != expected:
                    raise RuntimeError(
                        f"Expected {expected} bytes, wrote {written}"
                    )
            os.replace(temporary_path, part_path)
            return index, expected, False
        except Exception as error:
            print(
                f"retry chunk={index} attempt={attempt} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            if temporary_path.exists():
                temporary_path.unlink()
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Chunk {index} failed after {MAX_RETRIES} attempts")


def main():
    TARGET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PARTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    chunks = chunk_specs()
    completed = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = {
            executor.submit(download_chunk, chunk): chunk for chunk in chunks
        }
        for future in concurrent.futures.as_completed(futures):
            index, size, reused = future.result()
            completed += size
            if (index + 1) % 10 == 0 or completed == EXPECTED_BYTES:
                elapsed = max(time.time() - started, 1.0)
                speed = completed / elapsed / (1024 * 1024)
                percent = 100.0 * completed / EXPECTED_BYTES
                print(
                    f"completed_chunks={index + 1}/{len(chunks)} "
                    f"bytes={completed}/{EXPECTED_BYTES} "
                    f"percent={percent:.1f} speed_mib_s={speed:.2f}",
                    flush=True,
                )

    assembled_path = TARGET_DIRECTORY / "test.zip.assembled"
    with assembled_path.open("wb") as output:
        for index, _, _ in chunks:
            part_path = PARTS_DIRECTORY / f"{index:04d}.part"
            with part_path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
    if assembled_path.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"Assembled size mismatch: {assembled_path.stat().st_size}"
        )
    os.replace(assembled_path, TARGET_PATH)
    print(
        f"complete path={TARGET_PATH} bytes={TARGET_PATH.stat().st_size} "
        f"parts={math.ceil(EXPECTED_BYTES / CHUNK_BYTES)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
