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

df = pd.read_csv(
    os.path.join(BASE_PATH, "glm_fake_reviews_test.csv")
)

print("=" * 70)
print("CROSS-GENERATOR ROBUSTNESS TEST")
print("Training generator: DeepSeek")
print("External generator: GLM")
print("=" * 70)

print("\nExternal test reviews:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())

model = joblib.load(
    os.path.join(BASE_PATH, "best_fake_review_model.joblib")
)

tfidf = joblib.load(
    os.path.join(BASE_PATH, "tfidf_vectorizer.joblib")
)

X_external = tfidf.transform(df["text_"].astype(str))
y_true = df["label"]

predictions = model.predict(X_external)

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions, pos_label="CG")
recall = recall_score(y_true, predictions, pos_label="CG")
f1 = f1_score(y_true, predictions, pos_label="CG")

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
print(classification_report(y_true, predictions))

original_accuracy = 0.9772
original_f1 = 0.9770

print("\n" + "=" * 70)
print("ROBUSTNESS COMPARISON")
print("=" * 70)

print(f"Original DeepSeek test accuracy: {original_accuracy:.4f}")
print(f"GLM external accuracy: {accuracy:.4f}")
print(f"Accuracy change: {accuracy - original_accuracy:.4f}")

print(f"\nOriginal DeepSeek F1: {original_f1:.4f}")
print(f"GLM external F1: {f1:.4f}")
print(f"F1 change: {f1 - original_f1:.4f}")

df["prediction"] = predictions
df["correct"] = df["label"] == df["prediction"]

output_path = os.path.join(
    BASE_PATH,
    "glm_cross_generator_predictions.csv"
)

df.to_csv(output_path, index=False)

print("\nSaved:")
print(output_path)

print("\nGLM cross-generator test completed successfully.")
