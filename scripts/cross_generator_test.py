import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load external LFM test dataset
# --------------------------------------------------

df = pd.read_csv(
    os.path.join(
        BASE_PATH,
        "lfm_fake_reviews_test.csv"
    )
)

print("=" * 70)
print("CROSS-GENERATOR ROBUSTNESS TEST")
print("Training generator: DeepSeek")
print("External generator: LFM2.5")
print("=" * 70)

print("\nExternal test reviews:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())

# --------------------------------------------------
# Load ORIGINAL model
# No retraining
# --------------------------------------------------

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

# --------------------------------------------------
# Transform using ORIGINAL DeepSeek-trained TF-IDF
# --------------------------------------------------

X_external = tfidf.transform(
    df["text_"].astype(str)
)

y_true = df["label"]

# --------------------------------------------------
# Predict
# --------------------------------------------------

predictions = model.predict(X_external)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions,
    pos_label="CG"
)

recall = recall_score(
    y_true,
    predictions,
    pos_label="CG"
)

f1 = f1_score(
    y_true,
    predictions,
    pos_label="CG"
)

cm = confusion_matrix(
    y_true,
    predictions,
    labels=["CG", "OR"]
)

print("\nRESULTS")
print("-" * 40)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nConfusion Matrix [CG, OR]:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        predictions
    )
)

# --------------------------------------------------
# Compare with original DeepSeek test result
# --------------------------------------------------

original_accuracy = 0.9772
original_f1 = 0.9770

print("\n" + "=" * 70)
print("ROBUSTNESS COMPARISON")
print("=" * 70)

print(
    f"Original DeepSeek test accuracy: "
    f"{original_accuracy:.4f}"
)

print(
    f"LFM external accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Accuracy change: "
    f"{accuracy - original_accuracy:.4f}"
)

print(
    f"\nOriginal DeepSeek F1: "
    f"{original_f1:.4f}"
)

print(
    f"LFM external F1: "
    f"{f1:.4f}"
)

print(
    f"F1 change: "
    f"{f1 - original_f1:.4f}"
)

# Save predictions
df["prediction"] = predictions
df["correct"] = df["label"] == df["prediction"]

df.to_csv(
    os.path.join(
        BASE_PATH,
        "lfm_cross_generator_predictions.csv"
    ),
    index=False
)

print(
    "\nSaved: "
    "lfm_cross_generator_predictions.csv"
)

print("\nCross-generator test completed.")