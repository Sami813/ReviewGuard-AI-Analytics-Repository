import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Load test data
df = pd.read_csv(
    os.path.join(
        BASE_PATH,
        "modern_fake_reviews_test.csv"
    )
)

# Load trained model and TF-IDF vectorizer
model = joblib.load(
    os.path.join(
        BASE_PATH,
        "best_fake_review_model.joblib"
    )
)

tfidf = joblib.load(
    os.path.join(
        BASE_PATH,
        "tfidf_vectorizer.joblib"
    )
)

# Transform test reviews
X_test = tfidf.transform(
    df["text_"].astype(str)
)

# Predictions
df["prediction"] = model.predict(X_test)

# Correct / incorrect
df["correct"] = (
    df["label"] == df["prediction"]
)

# Review length
df["word_count"] = (
    df["text_"]
    .astype(str)
    .str.split()
    .str.len()
)

print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print("\nOverall correct predictions:")
print(df["correct"].value_counts())

print("\nOverall error rate:")
print(round((~df["correct"]).mean() * 100, 2), "%")

# ------------------------------------------------
# CATEGORY PERFORMANCE
# ------------------------------------------------

category_results = []

for category, group in df.groupby("category"):

    accuracy = accuracy_score(
        group["label"],
        group["prediction"]
    )

    f1 = f1_score(
        group["label"],
        group["prediction"],
        pos_label="CG"
    )

    category_results.append({
        "Category": category,
        "Reviews": len(group),
        "Accuracy": accuracy,
        "F1_CG": f1
    })

category_df = pd.DataFrame(
    category_results
).sort_values(
    "Accuracy",
    ascending=False
)

print("\n" + "=" * 70)
print("PERFORMANCE BY PRODUCT CATEGORY")
print("=" * 70)

print(category_df.to_string(index=False))

# ------------------------------------------------
# RATING PERFORMANCE
# ------------------------------------------------

rating_results = []

for rating, group in df.groupby("rating"):

    accuracy = accuracy_score(
        group["label"],
        group["prediction"]
    )

    f1 = f1_score(
        group["label"],
        group["prediction"],
        pos_label="CG"
    )

    rating_results.append({
        "Rating": rating,
        "Reviews": len(group),
        "Accuracy": accuracy,
        "F1_CG": f1
    })

rating_df = pd.DataFrame(
    rating_results
).sort_values("Rating")

print("\n" + "=" * 70)
print("PERFORMANCE BY RATING")
print("=" * 70)

print(rating_df.to_string(index=False))

# ------------------------------------------------
# REVIEW LENGTH GROUPS
# ------------------------------------------------

df["length_group"] = pd.cut(
    df["word_count"],
    bins=[0, 25, 50, 100, float("inf")],
    labels=[
        "Very Short (1-25)",
        "Short (26-50)",
        "Medium (51-100)",
        "Long (101+)"
    ]
)

length_results = []

for length, group in df.groupby(
    "length_group",
    observed=True
):

    accuracy = accuracy_score(
        group["label"],
        group["prediction"]
    )

    f1 = f1_score(
        group["label"],
        group["prediction"],
        pos_label="CG"
    )

    length_results.append({
        "Length_Group": length,
        "Reviews": len(group),
        "Accuracy": accuracy,
        "F1_CG": f1
    })

length_df = pd.DataFrame(length_results)

print("\n" + "=" * 70)
print("PERFORMANCE BY REVIEW LENGTH")
print("=" * 70)

print(length_df.to_string(index=False))

# ------------------------------------------------
# MISCLASSIFIED REVIEWS
# ------------------------------------------------

errors = df[
    df["label"] != df["prediction"]
].copy()

print("\nTotal misclassified:", len(errors))

print("\nTrue labels of errors:")
print(errors["label"].value_counts())

# Save results
category_df.to_csv(
    os.path.join(
        BASE_PATH,
        "performance_by_category.csv"
    ),
    index=False
)

rating_df.to_csv(
    os.path.join(
        BASE_PATH,
        "performance_by_rating.csv"
    ),
    index=False
)

length_df.to_csv(
    os.path.join(
        BASE_PATH,
        "performance_by_review_length.csv"
    ),
    index=False
)

errors.to_csv(
    os.path.join(
        BASE_PATH,
        "misclassified_reviews.csv"
    ),
    index=False
)

print("\nFiles saved:")
print("performance_by_category.csv")
print("performance_by_rating.csv")
print("performance_by_review_length.csv")
print("misclassified_reviews.csv")

print("\nError analysis completed successfully.")