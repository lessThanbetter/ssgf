"""Fetch selected Git blobs through one ``git cat-file --batch`` process."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("unexpected end of git cat-file stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--object",
        action="append",
        required=True,
        metavar="SHA=OUTPUT",
    )
    args = parser.parse_args()

    specs = []
    for item in args.object:
        sha, output = item.split("=", 1)
        specs.append((sha, Path(output)))

    process = subprocess.Popen(
        ["git", "-C", args.repo, "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    for sha, _ in specs:
        process.stdin.write((sha + "\n").encode("ascii"))
    process.stdin.close()

    for sha, output in specs:
        header = process.stdout.readline().decode("ascii").strip().split()
        if len(header) != 3 or header[1] != "blob":
            raise RuntimeError(f"cannot read {sha}: {header}")
        payload = read_exact(process.stdout, int(header[2]))
        if process.stdout.read(1) != b"\n":
            raise RuntimeError(f"missing object terminator for {sha}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)

    if process.wait() != 0:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
