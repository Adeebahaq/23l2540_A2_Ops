from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path("reports/splits")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/audit.csv")
df = df[~df["corrupt"] & ~df["is_duplicate"]].copy()      # drop corrupt files and the extra duplicate
df["label"] = (df["class"] == "AiArtData").astype(int)     # 1 = AI-generated, 0 = real

# 70% train, 15% val, 15% test, stratified so each split keeps the class ratio
train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

cols = ["path", "class", "label", "sha256"]
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[cols].to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(part), part["label"].value_counts().to_dict())

# safety check: no file appears in two splits
assert not (set(train.sha256) & set(val.sha256)), "leak train/val"
assert not (set(train.sha256) & set(test.sha256)), "leak train/test"
assert not (set(val.sha256) & set(test.sha256)), "leak val/test"
print("No overlap between splits.")