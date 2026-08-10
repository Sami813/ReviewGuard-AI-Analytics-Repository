# ReviewGuard AI Analytics

**Machine Learning-Based Fake Product Review Detection for E-Commerce Platforms**

ReviewGuard AI Analytics is an MSc research prototype for screening e-commerce product reviews as:

- **OR** — Original / human-written
- **CG** — Computer-generated

## Repository structure

```text
ReviewGuard-AI-Analytics/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   ├── best_fake_review_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── final_multigenerator_model.joblib
│   └── final_multigenerator_tfidf.joblib
├── data/
│   ├── modern_fake_reviews_train.csv
│   ├── modern_fake_reviews_validation.csv
│   ├── modern_fake_reviews_test.csv
│   ├── lfm_fake_reviews_train.csv
│   ├── lfm_fake_reviews_validation.csv
│   ├── lfm_fake_reviews_test.csv
│   ├── glm_fake_reviews_train.csv
│   ├── glm_fake_reviews_validation.csv
│   └── glm_fake_reviews_test.csv
├── results/
│   ├── baseline_model_results.csv
│   ├── controlled_cross_generator_comparison.csv
│   ├── leave_one_generator_out_results.csv
│   ├── performance_by_category.csv
│   ├── performance_by_rating.csv
│   ├── performance_by_review_length.csv
│   ├── misclassified_reviews.csv
│   ├── final_deployment_model_summary.csv
│   ├── leave_one_generator_overlap_report.csv
│   └── dashboard_test_sample.csv
├── scripts/
│   ├── train_baseline_models.py
│   ├── dataset_quality_check.py
│   ├── error_analysis.py
│   ├── controlled_cross_generator_comparison.py
│   ├── cross_dataset_overlap_check.py
│   ├── cross_generator_test.py
│   ├── glm_cross_generator_test.py
│   ├── leave_one_generator_out_training.py
│   ├── leave_one_generator_overlap_check.py
│   ├── train_final_deployment_model.py
│   └── create_dissertation_figures.py
└── figures/
    └── dissertation figures
```

## Key results

- Primary labelled dataset: **40,424 reviews**
- TF-IDF feature space: **50,000 features**
- Selected baseline model: **Multinomial Naive Bayes**
- Held-out test accuracy: **97.72%**
- CG F1-score: **97.70%**
- Controlled cross-generator average CG F1: **92.09%**
- Generator sources: **DeepSeek, LFM and GLM**

## Run locally

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the dashboard:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The repository version uses relative project folders, so it does not depend on `D:\Ahad Data Set`.

## Responsible interpretation

A CG prediction is a screening output, not proof of fraudulent intent. The dashboard is an MSc research prototype rather than a production moderation system.

## Dataset note

Before making the repository public, confirm that the dataset licences allow redistribution of the raw CSV files. If redistribution is restricted, keep the code/results public and replace the raw datasets with source/download instructions.
