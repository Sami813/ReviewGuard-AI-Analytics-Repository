import os
import time
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# 1. Paths
# ---------------------------------------------------

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(
    BASE_PATH,
    "modern_fake_reviews_train.csv"
)

validation_path = os.path.join(
    BASE_PATH,
    "modern_fake_reviews_validation.csv"
)

test_path = os.path.join(
    BASE_PATH,
    "modern_fake_reviews_test.csv"
)

# ---------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)

print("Train:", train_df.shape)
print("Validation:", validation_df.shape)
print("Test:", test_df.shape)

# ---------------------------------------------------
# 3. Remove duplicate review from training data
# ---------------------------------------------------

before = len(train_df)

train_df = train_df.drop_duplicates(
    subset=["text_"]
).reset_index(drop=True)

after = len(train_df)

print(
    f"Removed {before - after} duplicate review(s) "
    f"from training data."
)

# ---------------------------------------------------
# 4. Prepare text and labels
# ---------------------------------------------------

X_train_text = train_df["text_"].astype(str)
y_train = train_df["label"]

X_val_text = validation_df["text_"].astype(str)
y_val = validation_df["label"]

X_test_text = test_df["text_"].astype(str)
y_test = test_df["label"]

# ---------------------------------------------------
# 5. TF-IDF feature extraction
# ---------------------------------------------------

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(X_train_text)
X_val = tfidf.transform(X_val_text)
X_test = tfidf.transform(X_test_text)

print("Training feature matrix:", X_train.shape)
print("Validation feature matrix:", X_val.shape)
print("Test feature matrix:", X_test.shape)

# ---------------------------------------------------
# 6. Models
# ---------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Support Vector Machine": LinearSVC(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}

# ---------------------------------------------------
# 7. Train and evaluate on validation set
# ---------------------------------------------------

results = []
trained_models = {}

print("\n" + "=" * 70)
print("BASELINE MODEL TRAINING")
print("=" * 70)

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val,
        predictions,
        pos_label="CG"
    )

    recall = recall_score(
        y_val,
        predictions,
        pos_label="CG"
    )

    f1 = f1_score(
        y_val,
        predictions,
        pos_label="CG"
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=["CG", "OR"]
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision_CG": precision,
        "Recall_CG": recall,
        "F1_CG": f1,
        "Training_Time_Seconds": training_time
    })

    trained_models[model_name] = model

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    print("\nConfusion Matrix [CG, OR]:")
    print(cm)

# ---------------------------------------------------
# 8. Compare models
# ---------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_CG",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON — VALIDATION SET")
print("=" * 70)

print(results_df.to_string(index=False))

results_path = os.path.join(
    BASE_PATH,
    "baseline_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

# ---------------------------------------------------
# 9. Select best model using validation F1
# ---------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest validation model:")
print(best_model_name)

# ---------------------------------------------------
# 10. Final test evaluation
# ---------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    pos_label="CG"
)

test_recall = recall_score(
    y_test,
    test_predictions,
    pos_label="CG"
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    pos_label="CG"
)

test_cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=["CG", "OR"]
)

print("Best model:", best_model_name)
print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1-score:  {test_f1:.4f}")

print("\nTest Confusion Matrix [CG, OR]:")
print(test_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions
    )
)

# ---------------------------------------------------
# 11. Save final model and vectorizer
# ---------------------------------------------------

joblib.dump(
    best_model,
    os.path.join(
        BASE_PATH,
        "best_fake_review_model.joblib"
    )
)

joblib.dump(
    tfidf,
    os.path.join(
        BASE_PATH,
        "tfidf_vectorizer.joblib"
    )
)

print("\nFiles saved:")
print("baseline_model_results.csv")
print("best_fake_review_model.joblib")
print("tfidf_vectorizer.joblib")

print("\nTraining completed successfully.")