import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load data
# --------------------------------------------------

deep_train = pd.read_csv(os.path.join(BASE, "modern_fake_reviews_train.csv"))
deep_test = pd.read_csv(os.path.join(BASE, "modern_fake_reviews_test.csv"))
lfm_test = pd.read_csv(os.path.join(BASE, "lfm_fake_reviews_test.csv"))
glm_test = pd.read_csv(os.path.join(BASE, "glm_fake_reviews_test.csv"))

# Load the ORIGINAL DeepSeek-trained model and vectorizer
model = joblib.load(os.path.join(BASE, "best_fake_review_model.joblib"))
tfidf = joblib.load(os.path.join(BASE, "tfidf_vectorizer.joblib"))

# --------------------------------------------------
# Build one clean OR reference set
# --------------------------------------------------

train_texts = set(deep_train["text_"].dropna().astype(str))

clean_or = deep_test[deep_test["label"] == "OR"].copy()
clean_or = clean_or[~clean_or["text_"].astype(str).isin(train_texts)].copy()
clean_or = clean_or.drop_duplicates(subset=["text_"]).reset_index(drop=True)

print("=" * 78)
print("CONTROLLED CROSS-GENERATOR COMPARISON")
print("=" * 78)
print(f"\nClean OR reference reviews: {len(clean_or)}")
print("The same unseen OR reviews will be used for every generator comparison.")

# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_generator(generator_name, source_df):
    cg = source_df[source_df["label"] == "CG"].copy()

    before = len(cg)

    # Remove any CG text that appeared in DeepSeek training data
    cg = cg[~cg["text_"].astype(str).isin(train_texts)].copy()
    cg = cg.drop_duplicates(subset=["text_"]).reset_index(drop=True)

    removed = before - len(cg)

    evaluation_df = pd.concat(
        [cg, clean_or],
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    X = tfidf.transform(evaluation_df["text_"].astype(str))
    y_true = evaluation_df["label"]

    predictions = model.predict(X)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, pos_label="CG")
    recall = recall_score(y_true, predictions, pos_label="CG")
    f1 = f1_score(y_true, predictions, pos_label="CG")

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=["CG", "OR"]
    )

    print("\n" + "-" * 78)
    print(f"GENERATOR: {generator_name}")
    print("-" * 78)
    print(f"CG reviews before leakage filtering: {before}")
    print(f"CG reviews removed for train overlap/duplicates: {removed}")
    print(f"CG reviews evaluated: {len(cg)}")
    print(f"OR reviews evaluated: {len(clean_or)}")
    print(f"Total evaluation reviews: {len(evaluation_df)}")

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix [CG, OR]:")
    print(cm)

    evaluation_df["prediction"] = predictions
    evaluation_df["correct"] = evaluation_df["label"] == evaluation_df["prediction"]
    evaluation_df["generator_test"] = generator_name

    safe_name = generator_name.lower().replace(" ", "_")
    evaluation_df.to_csv(
        os.path.join(
            BASE,
            f"controlled_{safe_name}_predictions.csv"
        ),
        index=False
    )

    return {
        "Generator": generator_name,
        "CG_Reviews": len(cg),
        "OR_Reviews": len(clean_or),
        "Total_Reviews": len(evaluation_df),
        "Accuracy": accuracy,
        "Precision_CG": precision,
        "Recall_CG": recall,
        "F1_CG": f1,
        "CG_Correct": int(cm[0, 0]),
        "CG_Missed_as_OR": int(cm[0, 1]),
        "OR_Flagged_as_CG": int(cm[1, 0]),
        "OR_Correct": int(cm[1, 1])
    }

# --------------------------------------------------
# Run controlled comparisons
# --------------------------------------------------

results = []

results.append(
    evaluate_generator(
        "DeepSeek",
        deep_test
    )
)

results.append(
    evaluate_generator(
        "LFM",
        lfm_test
    )
)

results.append(
    evaluate_generator(
        "GLM",
        glm_test
    )
)

summary = pd.DataFrame(results)

print("\n" + "=" * 78)
print("FINAL CONTROLLED COMPARISON")
print("=" * 78)

print(
    summary[
        [
            "Generator",
            "Accuracy",
            "Precision_CG",
            "Recall_CG",
            "F1_CG",
            "CG_Missed_as_OR",
            "OR_Flagged_as_CG"
        ]
    ].to_string(index=False)
)

summary_path = os.path.join(
    BASE,
    "controlled_cross_generator_comparison.csv"
)

summary.to_csv(summary_path, index=False)

print("\nSaved:")
print(summary_path)
print("controlled_deepseek_predictions.csv")
print("controlled_lfm_predictions.csv")
print("controlled_glm_predictions.csv")

print("\nControlled comparison completed successfully.")
