import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "deep_train": "modern_fake_reviews_train.csv",
    "deep_val": "modern_fake_reviews_validation.csv",
    "deep_test": "modern_fake_reviews_test.csv",
    "lfm_train": "lfm_fake_reviews_train.csv",
    "lfm_val": "lfm_fake_reviews_validation.csv",
    "lfm_test": "lfm_fake_reviews_test.csv",
    "glm_train": "glm_fake_reviews_train.csv",
    "glm_val": "glm_fake_reviews_validation.csv",
    "glm_test": "glm_fake_reviews_test.csv",
}

data = {
    name: pd.read_csv(os.path.join(BASE, filename))
    for name, filename in FILES.items()
}

def text_set(df, label=None):
    if label is not None:
        df = df[df["label"] == label]
    return set(df["text_"].dropna().astype(str))

def overlap(a, b, label=None):
    return len(text_set(data[a], label) & text_set(data[b], label))

rows = []

def add_check(check_name, a, b, label):
    count = overlap(a, b, label)
    rows.append({
        "Check": check_name,
        "Dataset_A": a,
        "Dataset_B": b,
        "Label": label,
        "Overlap_Count": count
    })
    return count

print("=" * 82)
print("LEAVE-ONE-GENERATOR-OUT LEAKAGE CHECK")
print("=" * 82)

print("\nLoaded files:")
for name, df in data.items():
    print(f"{name:12s}: {len(df):5d} rows")

print("\n" + "=" * 82)
print("1. COMMON ORIGINAL-REVIEW TRAIN/TEST CHECK")
print("=" * 82)

count = add_check(
    "DeepSeek OR train vs common OR test",
    "deep_train", "deep_test", "OR"
)
print("DeepSeek TRAIN OR vs DeepSeek TEST OR:", count)

print("\n" + "=" * 82)
print("2. EXPERIMENT A: TRAIN DeepSeek + GLM, TEST LFM")
print("=" * 82)

a1 = add_check(
    "DeepSeek train CG vs LFM test CG",
    "deep_train", "lfm_test", "CG"
)
a2 = add_check(
    "GLM train CG vs LFM test CG",
    "glm_train", "lfm_test", "CG"
)
a3 = add_check(
    "GLM validation CG vs LFM test CG",
    "glm_val", "lfm_test", "CG"
)
a4 = add_check(
    "DeepSeek validation CG vs LFM test CG",
    "deep_val", "lfm_test", "CG"
)

print("DeepSeek TRAIN CG vs LFM TEST CG:", a1)
print("GLM TRAIN CG vs LFM TEST CG:", a2)
print("GLM VALIDATION CG vs LFM TEST CG:", a3)
print("DeepSeek VALIDATION CG vs LFM TEST CG:", a4)

print("\n" + "=" * 82)
print("3. EXPERIMENT B: TRAIN DeepSeek + LFM, TEST GLM")
print("=" * 82)

b1 = add_check(
    "DeepSeek train CG vs GLM test CG",
    "deep_train", "glm_test", "CG"
)
b2 = add_check(
    "LFM train CG vs GLM test CG",
    "lfm_train", "glm_test", "CG"
)
b3 = add_check(
    "LFM validation CG vs GLM test CG",
    "lfm_val", "glm_test", "CG"
)
b4 = add_check(
    "DeepSeek validation CG vs GLM test CG",
    "deep_val", "glm_test", "CG"
)

print("DeepSeek TRAIN CG vs GLM TEST CG:", b1)
print("LFM TRAIN CG vs GLM TEST CG:", b2)
print("LFM VALIDATION CG vs GLM TEST CG:", b3)
print("DeepSeek VALIDATION CG vs GLM TEST CG:", b4)

print("\n" + "=" * 82)
print("4. DUPLICATION BETWEEN GENERATOR TRAINING CG SETS")
print("=" * 82)

c1 = add_check(
    "DeepSeek train CG vs LFM train CG",
    "deep_train", "lfm_train", "CG"
)
c2 = add_check(
    "DeepSeek train CG vs GLM train CG",
    "deep_train", "glm_train", "CG"
)
c3 = add_check(
    "LFM train CG vs GLM train CG",
    "lfm_train", "glm_train", "CG"
)

print("DeepSeek TRAIN CG vs LFM TRAIN CG:", c1)
print("DeepSeek TRAIN CG vs GLM TRAIN CG:", c2)
print("LFM TRAIN CG vs GLM TRAIN CG:", c3)

print("\n" + "=" * 82)
print("5. WITHIN-FILE DUPLICATES")
print("=" * 82)

for name, df in data.items():
    dup_all = df.duplicated(subset=["text_"]).sum()
    dup_cg = df[df["label"] == "CG"].duplicated(subset=["text_"]).sum()
    dup_or = df[df["label"] == "OR"].duplicated(subset=["text_"]).sum()
    print(
        f"{name:12s}: all={dup_all:3d}, CG={dup_cg:3d}, OR={dup_or:3d}"
    )

report = pd.DataFrame(rows)
report_path = os.path.join(
    BASE,
    "leave_one_generator_overlap_report.csv"
)
report.to_csv(report_path, index=False)

critical_cols = report[
    report["Check"].str.contains(
        "test CG|common OR test",
        case=False,
        regex=True
    )
]

critical_overlap = critical_cols["Overlap_Count"].sum()

print("\n" + "=" * 82)
print("SUMMARY")
print("=" * 82)

if critical_overlap == 0:
    print("PASS: No critical train/validation-to-held-out-test overlap was found.")
    print("The leave-one-generator-out experiments can proceed.")
else:
    print("WARNING: Critical overlap was detected.")
    print("Do not train the leave-one-generator-out models until this is cleaned.")

print("\nSaved:")
print(report_path)

print("\nLeakage check completed successfully.")
