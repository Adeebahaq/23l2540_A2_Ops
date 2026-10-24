import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data/raw", help="folder with one sub-folder per class")
parser.add_argument("--out", default="reports/audit.csv")
args = parser.parse_args()

root = Path(args.root)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


rows = []
for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for f in sorted(class_dir.rglob("*")):
        if not f.is_file():
            continue
        row = {
            "path": f.as_posix(),
            "class": class_dir.name,
            "file_size": f.stat().st_size,
            "sha256": sha256_of(f),
            "width": None, "height": None, "mode": None, "format": None,
            "corrupt": False, "error": "",
        }
        try:
            with Image.open(f) as im:      # first pass: structural check
                im.verify()
            with Image.open(f) as im:      # verify() invalidates the file, so reopen
                im.load()                  # forces full decode
                row["width"], row["height"] = im.size
                row["mode"], row["format"] = im.mode, im.format
        except Exception as e:
            row["corrupt"] = True
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

df = pd.DataFrame(rows)

# Duplicates: same hash = byte-identical file
df["dup_group_size"] = df.groupby("sha256")["sha256"].transform("count")
df["is_duplicate"] = df["sha256"].duplicated(keep="first")   # extra copies only
df["dup_across_classes"] = df.groupby("sha256")["class"].transform("nunique") > 1

df.to_csv(args.out, index=False)

ok = df[~df["corrupt"]]
print("\n=== Images per class ===")
print(df.groupby("class").size().to_string())

print("\n=== Readable images: min / median / max ===")
for col in ["width", "height", "file_size"]:
    stats = ok.groupby("class")[col].agg(["min", "median", "max"])
    print(f"\n{col}:\n{stats.to_string()}")

print("\n=== Colour modes ===")
print(ok.groupby(["class", "mode"]).size().to_string())
print("\n=== Formats ===")
print(ok.groupby(["class", "format"]).size().to_string())

print("\n=== Problems ===")
print("Corrupt files:", int(df["corrupt"].sum()))
print("Duplicate extra copies:", int(df["is_duplicate"].sum()))
print("Files in duplicate groups across DIFFERENT classes:", int(df["dup_across_classes"].sum()))
print(f"\nFull table saved to {args.out}")