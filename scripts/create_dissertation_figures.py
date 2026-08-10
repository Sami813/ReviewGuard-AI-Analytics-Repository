import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "dissertation_figures")
os.makedirs(OUT, exist_ok=True)

def save_current(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# Figure 1: Baseline model comparison
# --------------------------------------------------
baseline = pd.read_csv(os.path.join(BASE, "baseline_model_results.csv"))
baseline = baseline.sort_values("F1_CG", ascending=False)

plt.figure(figsize=(9, 5))
x = range(len(baseline))
width = 0.25

plt.bar([i - width for i in x], baseline["Accuracy"], width=width, label="Accuracy")
plt.bar(x, baseline["Recall_CG"], width=width, label="Recall (CG)")
plt.bar([i + width for i in x], baseline["F1_CG"], width=width, label="F1-score (CG)")

plt.xticks(list(x), baseline["Model"], rotation=15, ha="right")
plt.ylim(0.80, 1.00)
plt.ylabel("Score")
plt.title("Baseline Model Performance on the Validation Set")
plt.legend()
save_current("Figure_1_Baseline_Model_Comparison.png")

# --------------------------------------------------
# Figure 2: Baseline training time
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(baseline["Model"], baseline["Training_Time_Seconds"])
plt.xticks(rotation=15, ha="right")
plt.ylabel("Training Time (seconds)")
plt.title("Baseline Model Training Time")
save_current("Figure_2_Baseline_Training_Time.png")

# --------------------------------------------------
# Figure 3: Controlled cross-generator comparison
# --------------------------------------------------
controlled = pd.read_csv(os.path.join(BASE, "controlled_cross_generator_comparison.csv"))

plt.figure(figsize=(8, 5))
x = range(len(controlled))
width = 0.25

plt.bar([i - width for i in x], controlled["Accuracy"], width=width, label="Accuracy")
plt.bar(x, controlled["Recall_CG"], width=width, label="Recall (CG)")
plt.bar([i + width for i in x], controlled["F1_CG"], width=width, label="F1-score (CG)")

plt.xticks(list(x), controlled["Generator"])
plt.ylim(0.65, 1.00)
plt.ylabel("Score")
plt.title("Controlled Cross-Generator Robustness")
plt.legend()
save_current("Figure_3_Controlled_Cross_Generator_Robustness.png")

# --------------------------------------------------
# Figure 4: Leave-one-generator-out comparison
# --------------------------------------------------
logo = pd.read_csv(os.path.join(BASE, "leave_one_generator_out_results.csv"))

labels = [
    f'{row["Training_Generators"]}\n→ {row["Held_Out_Generator"]}'
    for _, row in logo.iterrows()
]

plt.figure(figsize=(10, 6))
x = range(len(logo))
width = 0.25

plt.bar([i - width for i in x], logo["Accuracy"], width=width, label="Accuracy")
plt.bar(x, logo["Recall_CG"], width=width, label="Recall (CG)")
plt.bar([i + width for i in x], logo["F1_CG"], width=width, label="F1-score (CG)")

plt.xticks(list(x), labels)
plt.ylim(0.70, 1.00)
plt.ylabel("Score")
plt.title("Leave-One-Generator-Out Robustness")
plt.legend()
save_current("Figure_4_Leave_One_Generator_Out.png")

# --------------------------------------------------
# Figure 5: Performance by review length
# --------------------------------------------------
length_df = pd.read_csv(os.path.join(BASE, "performance_by_review_length.csv"))

plt.figure(figsize=(9, 5))
plt.bar(length_df["Length_Group"], length_df["Accuracy"])
plt.ylim(0.85, 1.00)
plt.ylabel("Accuracy")
plt.xlabel("Review Length Group")
plt.title("Naive Bayes Performance by Review Length")
plt.xticks(rotation=15, ha="right")
save_current("Figure_5_Performance_By_Review_Length.png")

# --------------------------------------------------
# Figure 6: Performance by product category
# --------------------------------------------------
category_df = pd.read_csv(os.path.join(BASE, "performance_by_category.csv"))
category_df = category_df.sort_values("Accuracy")

plt.figure(figsize=(10, 6))
plt.barh(category_df["Category"], category_df["Accuracy"])
plt.xlim(0.85, 1.00)
plt.xlabel("Accuracy")
plt.ylabel("Product Category")
plt.title("Naive Bayes Performance by Product Category")
save_current("Figure_6_Performance_By_Category.png")

print("Figures created successfully in:")
print(OUT)
