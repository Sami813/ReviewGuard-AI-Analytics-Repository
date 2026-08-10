import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

deep_train = pd.read_csv(os.path.join(BASE, "modern_fake_reviews_train.csv"))
deep_test = pd.read_csv(os.path.join(BASE, "modern_fake_reviews_test.csv"))

lfm_test = pd.read_csv(os.path.join(BASE, "lfm_fake_reviews_test.csv"))
glm_test = pd.read_csv(os.path.join(BASE, "glm_fake_reviews_test.csv"))

def texts(df, label=None):
    if label is not None:
        df = df[df["label"] == label]
    return set(df["text_"].dropna().astype(str))

print("=" * 70)
print("CROSS-DATASET OVERLAP CHECK")
print("=" * 70)

print("\nDeepSeek TRAIN vs LFM TEST")
print("All reviews:", len(texts(deep_train) & texts(lfm_test)))
print("OR reviews:", len(texts(deep_train, "OR") & texts(lfm_test, "OR")))
print("CG reviews:", len(texts(deep_train, "CG") & texts(lfm_test, "CG")))

print("\nDeepSeek TRAIN vs GLM TEST")
print("All reviews:", len(texts(deep_train) & texts(glm_test)))
print("OR reviews:", len(texts(deep_train, "OR") & texts(glm_test, "OR")))
print("CG reviews:", len(texts(deep_train, "CG") & texts(glm_test, "CG")))

print("\nDeepSeek TEST vs LFM TEST")
print("All reviews:", len(texts(deep_test) & texts(lfm_test)))
print("OR reviews:", len(texts(deep_test, "OR") & texts(lfm_test, "OR")))
print("CG reviews:", len(texts(deep_test, "CG") & texts(lfm_test, "CG")))

print("\nDeepSeek TEST vs GLM TEST")
print("All reviews:", len(texts(deep_test) & texts(glm_test)))
print("OR reviews:", len(texts(deep_test, "OR") & texts(glm_test, "OR")))
print("CG reviews:", len(texts(deep_test, "CG") & texts(glm_test, "CG")))

print("\nLFM TEST vs GLM TEST")
print("All reviews:", len(texts(lfm_test) & texts(glm_test)))
print("OR reviews:", len(texts(lfm_test, "OR") & texts(glm_test, "OR")))
print("CG reviews:", len(texts(lfm_test, "CG") & texts(glm_test, "CG")))

print("\nOverlap check completed.")