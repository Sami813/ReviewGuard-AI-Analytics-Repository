import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

deep_train = pd.read_csv(os.path.join(BASE, "modern_fake_reviews_train.csv"))
deep_test = pd.read_csv(os.path.join(BASE, "modern_fake_reviews_test.csv"))

lfm_train = pd.read_csv(os.path.join(BASE, "lfm_fake_reviews_train.csv"))
lfm_test = pd.read_csv(os.path.join(BASE, "lfm_fake_reviews_test.csv"))

glm_train = pd.read_csv(os.path.join(BASE, "glm_fake_reviews_train.csv"))
glm_test = pd.read_csv(os.path.join(BASE, "glm_fake_reviews_test.csv"))

DATASETS = {
    "DeepSeek": {"train": deep_train, "test": deep_test},
    "LFM": {"train": lfm_train, "test": lfm_test},
    "GLM": {"train": glm_train, "test": glm_test},
}

# --------------------------------------------------
# Common OR pools
# --------------------------------------------------

# Use the same DeepSeek OR training pool in every experiment.
# This avoids repeated human reviews across companion datasets and
# makes the only changing factor the CG generator diversity.
or_train = (
    deep_train[deep_train["label"] == "OR"]
    .drop_duplicates(subset=["text_"])
    .reset_index(drop=True)
)

# Use the same unseen OR test pool for every held-out generator.
or_test = (
    deep_test[deep_test["label"] == "OR"]
    .drop_duplicates(subset=["text_"])
    .reset_index(drop=True)
)

train_or_texts = set(or_train["text_"].astype(str))
test_or_texts = set(or_test["text_"].astype(str))

if train_or_texts & test_or_texts:
    raise RuntimeError(
        "STOP: OR overlap found between training and common test pool."
    )

print("=" * 84)
print("LEAVE-ONE-GENERATOR-OUT TRAINING")
print("=" * 84)

print(f"\nCommon OR training reviews: {len(or_train)}")
print(f"Common OR test reviews:     {len(or_test)}")
print("OR train/test overlap:      0")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_cg(df):
    return (
        df[df["label"] == "CG"]
        .drop_duplicates(subset=["text_"])
        .reset_index(drop=True)
    )

def balanced_two_generator_cg(name_a, name_b, target_total):
    cg_a = get_cg(DATASETS[name_a]["train"])
    cg_b = get_cg(DATASETS[name_b]["train"])

    n_a = target_total // 2
    n_b = target_total - n_a

    if len(cg_a) < n_a or len(cg_b) < n_b:
        raise RuntimeError(
            f"Not enough CG rows to balance {name_a} + {name_b}."
        )

    sample_a = cg_a.sample(n=n_a, random_state=RANDOM_STATE)
    sample_b = cg_b.sample(n=n_b, random_state=RANDOM_STATE + 1)

    combined = pd.concat(
        [sample_a, sample_b],
        ignore_index=True
    ).drop_duplicates(subset=["text_"])

    return combined.reset_index(drop=True), n_a, n_b

def run_experiment(train_gen_a, train_gen_b, held_out_gen):
    print("\n" + "=" * 84)
    print(
        f"TRAIN: {train_gen_a} + {train_gen_b}   |   "
        f"UNSEEN TEST GENERATOR: {held_out_gen}"
    )
    print("=" * 84)

    # --------------------------------------------------
    # Balanced training set
    # --------------------------------------------------

    target_cg = len(or_train)

    cg_train, n_a, n_b = balanced_two_generator_cg(
        train_gen_a,
        train_gen_b,
        target_cg
    )

    # Held-out CG test set
    heldout_cg = get_cg(DATASETS[held_out_gen]["test"])

    # Leakage guard: no exact held-out CG text may appear in training CG
    train_cg_texts = set(cg_train["text_"].astype(str))
    heldout_cg_texts = set(heldout_cg["text_"].astype(str))

    cg_overlap = len(train_cg_texts & heldout_cg_texts)

    print(f"\nCG sampled from {train_gen_a}: {n_a}")
    print(f"CG sampled from {train_gen_b}: {n_b}")
    print(f"Total CG training reviews:     {len(cg_train)}")
    print(f"Total OR training reviews:     {len(or_train)}")
    print(f"Held-out CG test reviews:      {len(heldout_cg)}")
    print(f"Common OR test reviews:        {len(or_test)}")
    print(f"CG train/test overlap:         {cg_overlap}")

    if cg_overlap != 0:
        raise RuntimeError(
            f"STOP: CG leakage detected for held-out generator {held_out_gen}."
        )

    train_df = pd.concat(
        [cg_train, or_train],
        ignore_index=True
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    test_df = pd.concat(
        [heldout_cg, or_test],
        ignore_index=True
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # --------------------------------------------------
    # TF-IDF: same configuration as baseline
    # --------------------------------------------------

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    X_train = tfidf.fit_transform(
        train_df["text_"].astype(str)
    )

    X_test = tfidf.transform(
        test_df["text_"].astype(str)
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    # --------------------------------------------------
    # Same lightweight classifier as baseline
    # --------------------------------------------------

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        pos_label="CG"
    )
    recall = recall_score(
        y_test,
        predictions,
        pos_label="CG"
    )
    f1 = f1_score(
        y_test,
        predictions,
        pos_label="CG"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["CG", "OR"]
    )

    print("\nRESULTS")
    print("-" * 44)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix [CG, OR]:")
    print(cm)

    # --------------------------------------------------
    # Save predictions and experiment model
    # --------------------------------------------------

    tag = (
        f"{train_gen_a.lower()}_{train_gen_b.lower()}"
        f"_test_{held_out_gen.lower()}"
    )

    test_df["prediction"] = predictions
    test_df["correct"] = (
        test_df["label"] == test_df["prediction"]
    )
    test_df["held_out_generator"] = held_out_gen

    prediction_file = os.path.join(
        BASE,
        f"logo_{tag}_predictions.csv"
    )

    model_file = os.path.join(
        BASE,
        f"logo_{tag}_model.joblib"
    )

    vectorizer_file = os.path.join(
        BASE,
        f"logo_{tag}_tfidf.joblib"
    )

    test_df.to_csv(prediction_file, index=False)
    joblib.dump(model, model_file)
    joblib.dump(tfidf, vectorizer_file)

    return {
        "Training_Generators": f"{train_gen_a} + {train_gen_b}",
        "Held_Out_Generator": held_out_gen,
        "Train_OR": len(or_train),
        "Train_CG": len(cg_train),
        "Test_OR": len(or_test),
        "Test_CG": len(heldout_cg),
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
# Run full three-fold leave-one-generator-out test
# --------------------------------------------------

results = []

results.append(
    run_experiment(
        "DeepSeek",
        "GLM",
        "LFM"
    )
)

results.append(
    run_experiment(
        "DeepSeek",
        "LFM",
        "GLM"
    )
)

results.append(
    run_experiment(
        "LFM",
        "GLM",
        "DeepSeek"
    )
)

summary = pd.DataFrame(results)

print("\n" + "=" * 84)
print("FINAL LEAVE-ONE-GENERATOR-OUT COMPARISON")
print("=" * 84)

print(
    summary[
        [
            "Training_Generators",
            "Held_Out_Generator",
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
    "leave_one_generator_out_results.csv"
)

summary.to_csv(summary_path, index=False)

print("\nSaved summary:")
print(summary_path)

print("\nLeave-one-generator-out experiment completed successfully.")
