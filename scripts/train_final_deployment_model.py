import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

BASE = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42

def load(name):
    return pd.read_csv(os.path.join(BASE, name))

deep_train = load("modern_fake_reviews_train.csv")
deep_val = load("modern_fake_reviews_validation.csv")

lfm_train = load("lfm_fake_reviews_train.csv")
lfm_val = load("lfm_fake_reviews_validation.csv")

glm_train = load("glm_fake_reviews_train.csv")
glm_val = load("glm_fake_reviews_validation.csv")

# --------------------------------------------------
# Original / human review pool
# Use one source to avoid duplicating shared OR reviews.
# --------------------------------------------------
or_pool = pd.concat([
    deep_train[deep_train["label"] == "OR"],
    deep_val[deep_val["label"] == "OR"]
], ignore_index=True)

or_pool = (
    or_pool
    .drop_duplicates(subset=["text_"])
    .reset_index(drop=True)
)

# --------------------------------------------------
# Computer-generated pools from all three generators
# --------------------------------------------------
def cg_pool(train_df, val_df):
    df = pd.concat([
        train_df[train_df["label"] == "CG"],
        val_df[val_df["label"] == "CG"]
    ], ignore_index=True)
    return df.drop_duplicates(subset=["text_"]).reset_index(drop=True)

deep_cg = cg_pool(deep_train, deep_val)
lfm_cg = cg_pool(lfm_train, lfm_val)
glm_cg = cg_pool(glm_train, glm_val)

# Balance the final deployment training set:
# total CG = total OR, with equal CG contribution from the three generators.
target_total_cg = len(or_pool)
n_each = target_total_cg // 3
remainder = target_total_cg - (n_each * 3)

deep_n = n_each + (1 if remainder > 0 else 0)
lfm_n = n_each + (1 if remainder > 1 else 0)
glm_n = n_each

deep_sample = deep_cg.sample(n=deep_n, random_state=RANDOM_STATE)
lfm_sample = lfm_cg.sample(n=lfm_n, random_state=RANDOM_STATE + 1)
glm_sample = glm_cg.sample(n=glm_n, random_state=RANDOM_STATE + 2)

cg_balanced = pd.concat(
    [deep_sample, lfm_sample, glm_sample],
    ignore_index=True
).drop_duplicates(subset=["text_"]).reset_index(drop=True)

# If deduplication changed the exact target by a tiny amount, trim OR to match.
final_n = min(len(or_pool), len(cg_balanced))
or_final = or_pool.sample(n=final_n, random_state=RANDOM_STATE)
cg_final = cg_balanced.sample(n=final_n, random_state=RANDOM_STATE)

train_df = pd.concat(
    [or_final, cg_final],
    ignore_index=True
).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("=" * 72)
print("FINAL MULTI-GENERATOR DEPLOYMENT MODEL")
print("=" * 72)
print("OR training reviews:", len(or_final))
print("CG training reviews:", len(cg_final))
print("Total training reviews:", len(train_df))
print("DeepSeek CG contribution:", deep_n)
print("LFM CG contribution:", lfm_n)
print("GLM CG contribution:", glm_n)

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(train_df["text_"].astype(str))
y_train = train_df["label"]

model = MultinomialNB()
model.fit(X_train, y_train)

model_path = os.path.join(BASE, "final_multigenerator_model.joblib")
tfidf_path = os.path.join(BASE, "final_multigenerator_tfidf.joblib")

joblib.dump(model, model_path)
joblib.dump(tfidf, tfidf_path)

summary = pd.DataFrame([{
    "OR_Training_Reviews": len(or_final),
    "CG_Training_Reviews": len(cg_final),
    "Total_Training_Reviews": len(train_df),
    "DeepSeek_CG_Contribution": deep_n,
    "LFM_CG_Contribution": lfm_n,
    "GLM_CG_Contribution": glm_n,
    "Feature_Count": X_train.shape[1],
    "Classifier": "Multinomial Naive Bayes"
}])

summary.to_csv(
    os.path.join(BASE, "final_deployment_model_summary.csv"),
    index=False
)

print("\nSaved:")
print(model_path)
print(tfidf_path)
print(os.path.join(BASE, "final_deployment_model_summary.csv"))
print("\nFinal deployment model trained successfully.")
print("Note: this model is for the prototype deployment stage.")
print("Use the earlier held-out experiments for unbiased dissertation evaluation.")
