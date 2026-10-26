import csv
import os
import random
import re
import sys
import time

import requests

BASE = "http://127.0.0.1:5000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4        # images per class
rows = list(csv.DictReader(open("reports/splits/test.csv", encoding="utf-8")))
random.seed(7)
picked = []
for lab in ("0", "1"):
    picked += random.sample([r for r in rows if r["label"] == lab], N)
random.shuffle(picked)

print(f"{'file':48} {'true':13} {'predicted':13} {'conf':>5} {'ms':>7}")
correct = 0
for r in picked:
    true = "AI-generated" if r["label"] == "1" else "Real"
    t0 = time.perf_counter()
    with open(r["path"], "rb") as fh:
        resp = requests.post(f"{BASE}/predict",
                             files={"image": (os.path.basename(r["path"]), fh)}, timeout=240)
    ms = (time.perf_counter() - t0) * 1000
    name = os.path.basename(r["path"])[:46]
    if resp.status_code != 200:
        print(f"{name:48} {true:13} HTTP {resp.status_code}")
        continue
    verdict = re.search(r'class="verdict (?:ai|real)">([^<]+)<', resp.text)
    conf = re.search(r"Confidence: <strong>(\d+)%", resp.text)
    pred = verdict.group(1) if verdict else "?"
    correct += pred == true
    print(f"{name:48} {true:13} {pred:13} {(conf.group(1) + '%') if conf else '?':>5} {ms:7.0f}")
print(f"\nagreement with ground truth: {correct}/{len(picked)}")