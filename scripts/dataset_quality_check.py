import pandas as pd
import os

base_path = os.path.dirname(os.path.abspath(__file__))

files = {
    "train": "modern_fake_reviews_train.csv",
    "validation": "modern_fake_reviews_validation.csv",
    "test": "modern_fake_reviews_test.csv"
}

datasets = {}

for split, filename in files.items():
    path = os.path.join(base_path, filename)
    datasets[split] = pd.read_csv(path)

print("=" * 70)
print("DATASET QUALITY ANALYSIS")
print("=" * 70)

for split, df in datasets.items():
    print(f"\n{split.upper()} SET")
    print("-" * 40)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nDuplicate reviews:")
    print(df.duplicated(subset=["text_"]).sum())

    print("\nCategories:", df["category"].nunique())
    print("Ratings:")
    print(df["rating"].value_counts().sort_index())

    df["word_count"] = (
        df["text_"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print("\nReview length statistics:")
    print(df["word_count"].describe())

    print("\nReview length by label:")
    print(
        df.groupby("label")["word_count"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )


print("\n" + "=" * 70)
print("CROSS-SPLIT DUPLICATE / LEAKAGE CHECK")
print("=" * 70)

train_text = set(datasets["train"]["text_"].dropna())
validation_text = set(datasets["validation"]["text_"].dropna())
test_text = set(datasets["test"]["text_"].dropna())

print(
    "Train vs Validation duplicates:",
    len(train_text.intersection(validation_text))
)

print(
    "Train vs Test duplicates:",
    len(train_text.intersection(test_text))
)

print(
    "Validation vs Test duplicates:",
    len(validation_text.intersection(test_text))
)


print("\n" + "=" * 70)
print("TRAINING SET CATEGORY DISTRIBUTION")
print("=" * 70)

print(datasets["train"]["category"].value_counts())


print("\n" + "=" * 70)
print("TRAINING SET RATING × LABEL")
print("=" * 70)

print(
    pd.crosstab(
        datasets["train"]["rating"],
        datasets["train"]["label"]
    )
)

print("\nAnalysis completed successfully.")