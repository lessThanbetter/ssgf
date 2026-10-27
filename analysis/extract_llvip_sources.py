"""Extract only the LLVIP test source pairs named by an analysis CSV.

The public LLVIP archive is about 4 GB, while the paper's detection analysis
uses 300 named test stems.  This script reads the remote ZIP directory, then
fetches only the compressed byte ranges for those visible/infrared JPEGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from remotezip import RemoteZip


LLVIP_URL = (
    "https://hf-mirror.com/datasets/jsonhash/LLVIP/resolve/main/LLVIP.zip"
)
USER_AGENT = "Mozilla/5.0"
LOCAL = threading.local()


def session() -> requests.Session:
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = USER_AGENT
    return LOCAL.session


def resolve_url(url: str) -> str:
    response = session().get(
        url,
        headers={"Range": "bytes=0-0"},
        stream=True,
        timeout=(30, 120),
    )
    response.raise_for_status()
    return response.url


def fetch_member(url: str, info, destination: Path, attempts: int = 5) -> Path:
    if destination.exists():
        return destination
    upper = (
        info.header_offset
        + 30
        + 65535
        + info.compress_size
        - 1
    )
    headers = {"Range": f"bytes={info.header_offset}-{upper}"}
    for attempt in range(1, attempts + 1):
        try:
            response = session().get(
                url, headers=headers, stream=True, timeout=(30, 180)
            )
            response.raise_for_status()
            payload = response.content
            if payload[:4] != b"PK\x03\x04":
                raise RuntimeError("missing ZIP local-file header")
            name_length, extra_length = struct.unpack_from("<HH", payload, 26)
            data_start = 30 + name_length + extra_length
            data_end = data_start + info.compress_size
            if data_end > len(payload):
                raise RuntimeError(
                    f"short ZIP payload {len(payload)} < {data_end}"
                )
            data = payload[data_start:data_end]
            if info.compress_type == 0:
                decoded = data
            elif info.compress_type == 8:
                decoded = zlib.decompress(data, -zlib.MAX_WBITS)
            else:
                raise RuntimeError(
                    f"unsupported ZIP compression {info.compress_type}"
                )
            if len(decoded) != info.file_size:
                raise RuntimeError(
                    f"decoded size {len(decoded)} != {info.file_size}"
                )
            if zlib.crc32(decoded) & 0xFFFFFFFF != info.CRC:
                raise RuntimeError("CRC mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(decoded)
            temporary.replace(destination)
            return destination
        except (requests.RequestException, OSError, RuntimeError) as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"{info.filename} failed after {attempts} attempts"
                ) from error
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def read_stems(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "stem" not in rows[0]:
        raise SystemExit(f"{path} must contain a 'stem' column")
    return [str(row["stem"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=LLVIP_URL)
    parser.add_argument("--stems", required=True, help="CSV with a stem column")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    stems = read_stems(Path(args.stems))
    if args.limit:
        stems = stems[: args.limit]
    wanted = set(stems)
    out = Path(args.out)
    print(f"[llvip-extract] reading remote ZIP directory for {len(stems)} stems",
          flush=True)
    with RemoteZip(args.url, support_suffix_range=True) as archive:
        members = {
            (info.filename, info.filename.split("/")[1]): info
            for info in archive.infolist()
            if (
                info.filename.startswith("LLVIP/visible/test/")
                or info.filename.startswith("LLVIP/infrared/test/")
            )
            and Path(info.filename).stem in wanted
        }
    if len(members) != 2 * len(stems):
        found = sorted({Path(name).stem for name, _ in members})
        missing = sorted(wanted - set(found))
        raise SystemExit(f"found {len(members)} files, expected {2 * len(stems)}; "
                         f"missing={missing[:10]}")

    final_url = resolve_url(args.url)
    jobs = []
    for (name, modality), info in members.items():
        destination = out / modality / f"{Path(name).stem}.jpg"
        jobs.append((info, destination))
    print(f"[llvip-extract] fetching {len(jobs)} source images", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as pool:
        futures = {
            pool.submit(fetch_member, final_url, info, destination): destination
            for info, destination in jobs
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"[llvip-extract] {completed}/{len(jobs)}", flush=True)

    manifest = {
        "url": args.url,
        "stems": stems,
        "files": {
            stem: {
                modality: str(out / modality / f"{stem}.jpg")
                for modality in ("visible", "infrared")
            }
            for stem in stems
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[llvip-extract] complete: {out}", flush=True)


if __name__ == "__main__":
    main()
