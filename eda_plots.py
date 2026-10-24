from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from PIL import Image

SEED = 42
PROJECT = "23l2540_A2_Ops"          # your roll number, appears on every figure
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/audit.csv")
df = df[~df["corrupt"]]                      # drop unreadable files (none here)
df = df[~df["is_duplicate"]]                 # drop the extra duplicate copy
print(df.groupby("class").size())

# ---------- 1. Sample grids: 12 random images per class ----------
for cls, g in df.groupby("class"):
    sample = g.sample(12, random_state=SEED)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((400, 400))         # shrink only for display
            ax.imshow(im)
        ax.set_title(f'{row["width"]}x{row["height"]}, {row["format"]}', fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Random samples: {cls}  [{PROJECT}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"samples_{cls}.png", dpi=120)
    plt.close(fig)

# ---------- 2. Histograms per class ----------
df["file_size_kb"] = df["file_size"] / 1024
metrics = [("width", "Width (pixels)"),
           ("height", "Height (pixels)"),
           ("file_size_kb", "File size (KB)")]
colors = {c: col for c, col in zip(sorted(df["class"].unique()), ["tab:blue", "tab:orange"])}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (col, label) in zip(axes, metrics):
    bins = np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 30)  # log bins: ranges are huge
    for cls, g in df.groupby("class"):
        ax.hist(g[col], bins=bins, alpha=0.55, label=cls, color=colors[cls])
        ax.axvline(g[col].median(), color=colors[cls], linestyle="--", linewidth=1)
    ax.set_xscale("log")

    # keep a few clean tick positions, drop the crowded minor labels
    lo, hi = df[col].min(), df[col].max()
    ticks = [t for t in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
             if lo * 0.8 <= t <= hi * 1.2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(NullFormatter())   # hides the overlapping labels

    ax.set_xlabel(label + "  (log scale)")
    ax.set_ylabel("Number of images")
    ax.legend()
fig.suptitle(f"Distributions per class (dashed line = median)  [{PROJECT}]")
fig.tight_layout()
fig.savefig(OUT / "histograms.png", dpi=120)
plt.close(fig)

print("Saved figures to", OUT)