"""Resolve the pinned official Chain archives and resume missing training downloads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import subprocess
import time

from e2.common import ROOT, read, sha, write

HERE = ROOT / "e4_chain"
OUT = HERE / "outputs"
RAW = ROOT / "bpp/datasets/libero"
CATALOG = read(HERE / "catalog.json")
FIRST, SECOND, DOUBLE, TEST = (
    "libero_goal_chain_" + suffix
    for suffix in (
        "firststep_view",
        "secondstep_view",
        "selected_inverse_view",
        "selected_view",
    )
)
TRAIN = (FIRST, SECOND, DOUBLE)


def verified(path, size, checksum):
    if not path.is_file() or path.stat().st_size != size:
        return False
    st = path.stat()
    stamp = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    downloaded = (
        ROOT
        / "bpp/.cache/tmp/bpp_download_hashes"
        / checksum
        / path.parent.name
        / (path.name + ".json")
    )
    marker = OUT / "verified" / path.parent.name / (path.name + ".json")
    if downloaded.is_file() and read(downloaded) == stamp:
        return True
    record = dict(stamp=stamp, sha256=checksum)
    if marker.is_file() and read(marker) == record:
        return True
    if sha(path) != checksum:
        raise ValueError(f"Archive checksum mismatch: {path}")
    write(marker, record)
    return True


def fetch(relative):
    size, checksum = CATALOG["files"][relative]
    path = RAW / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = OUT / "locks" / (relative + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if verified(path, size, checksum):
            return
        if path.exists():
            raise ValueError(f"Incomplete existing archive: {path}")
        stock = RAW / "libero_goal" / path.name
        if verified(stock, size, checksum):
            path.symlink_to(os.path.relpath(stock, path.parent))
            print("reused", relative, flush=True)
            return
        url = f"https://huggingface.co/datasets/{CATALOG['repo']}/resolve/{CATALOG['revision']}/demonstration_data/{relative}"
        part = path.with_suffix(".hdf5.e4.part")
        for attempt in range(32):
            if not part.exists() or part.stat().st_size < size:
                p = subprocess.run(
                    [
                        "curl",
                        *(["--http1.1"] if part.exists() or attempt >= 3 else []),
                        "-fsSL",
                        "-C",
                        "-",
                        "--connect-timeout",
                        "30",
                        "--speed-limit",
                        "4096",
                        "--speed-time",
                        "60",
                        "--max-time",
                        "900",
                        "-o",
                        str(part),
                        url + f"?download=true&resume={time.time_ns()}",
                    ],
                    capture_output=True,
                )
                if p.returncode:
                    print("retry", relative, attempt + 1, p.returncode, flush=True)
            if part.exists() and part.stat().st_size >= size:
                if part.stat().st_size == size and sha(part) == checksum:
                    part.replace(path)
                    print("verified", relative, flush=True)
                    return
                part.rename(part.with_suffix(f".bad-{time.time_ns()}"))
            time.sleep(min(30, 2 ** min(attempt, 5)))
    raise RuntimeError(f"Download incomplete: {relative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", nargs="+", choices=TRAIN, default=TRAIN)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    files = [r for r in CATALOG["files"] if r.split("/")[0] in args.views]
    print("Chain training archives in queue:", len(files), flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(fetch, files))
    print("Complete", flush=True)
