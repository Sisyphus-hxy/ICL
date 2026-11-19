import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/behavior_transfer"
OUT = ROOT / "e2/outputs"
E1 = ROOT / "contextflow_probe/e1"
B1 = ROOT / "bpp/experiments/e1"
OFFICIAL = {
    "cf": E1 / "outputs/shared/cf/official",
    "bpp": B1 / "outputs/shared/bpp/official",
}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, allow_nan=False)
    os.replace(f.name, path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(f.name, path)
