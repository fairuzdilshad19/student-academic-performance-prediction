import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Random state
RANDOM_STATE = 42

# Load dataset
url = "https://raw.githubusercontent.com/mibur1/psy111/main/book/statistics/4_Moderated_Reg/data/StressLevelDataset.csv"
df_raw = pd.read_csv(url)
df = df_raw.copy()

# Initial exploration
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Data cleaning
df = df.drop(columns=["blood_pressure", "headache"])

numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Feature engineering
df["mental_health_score"] = (
    df["depression"] + df["anxiety_level"]
) / 2

df["low_performance"] = (
    df["academic_performance"] <= 2
).astype(int)

df = df.drop(columns=["academic_performance"])

print("Final dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFirst 5 rows:")
print(df.head())

# Figure 1
counts = df["low_performance"].value_counts().sort_index()
values = [counts[0], counts[1]]

plt.figure(figsize=(7, 4))

plt.bar(
    ["High Performance", "Low Performance"],
    values,
    width=0.35
)

plt.title("Figure 1: Academic Performance Distribution")
plt.xlabel("Performance Group")
plt.ylabel("Number of Students")
plt.ylim(0, max(values) * 1.12)

for i, value in enumerate(values):
    plt.text(
        i,
        value + 5,
        str(value),
        ha="center"
    )

plt.tight_layout()

plt.savefig("figure1_academic_performance.png")

plt.show()

# Figure 2
stress = df.groupby(
    "low_performance"
)["stress_level"].mean()

plt.figure(figsize=(7, 4))

plt.bar(
    [0, 0.35],
    [stress[0], stress[1]],
    width=0.25,
    color=["steelblue", "lightcoral"]
)

plt.xticks(
    [0, 0.35],
    ["High Performance", "Low Performance"]
)

plt.title("Average Stress Level by Performance Group")
plt.xlabel("Performance Group")
plt.ylabel("Average Stress Level")
plt.ylim(0, max(stress) * 1.15)

plt.text(
    0,
    stress[0] + 0.04,
    f"{stress[0]:.2f}",
    ha="center"
)

plt.text(
    0.35,
    stress[1] + 0.04,
    f"{stress[1]:.2f}",
    ha="center"
)

plt.tight_layout()

plt.savefig("figure2_stress.png")

plt.show()

# Figure 3
sleep = df.groupby(
    "low_performance"
)["sleep_quality"].mean()

plt.figure(figsize=(7, 4))

plt.bar(
    [0, 0.35],
    [sleep[0], sleep[1]],
    width=0.25,
    color=["steelblue", "lightcoral"]
)

plt.xticks(
    [0, 0.35],
    ["High Performance", "Low Performance"]
)

plt.title("Average Sleep Quality by Performance Group")
plt.xlabel("Performance Group")
plt.ylabel("Average Sleep Quality")
plt.ylim(0, max(sleep) * 1.15)

plt.text(
    0,
    sleep[0] + 0.05,
    f"{sleep[0]:.2f}",
    ha="center"
)

plt.text(
    0.35,
    sleep[1] + 0.05,
    f"{sleep[1]:.2f}",
    ha="center"
)

plt.tight_layout()

plt.savefig("figure3_sleep_quality.png")

plt.show()

# Figure 4
mental_health = df.groupby(
    "low_performance"
)["mental_health_score"].mean()

plt.figure(figsize=(7, 4))

plt.bar(
    [0, 0.35],
    [mental_health[0], mental_health[1]],
    width=0.25,
    color=["steelblue", "lightcoral"]
)

plt.xticks(
    [0, 0.35],
    ["High Performance", "Low Performance"]
)

plt.title("Average Mental Health Score by Performance Group")
plt.xlabel("Performance Group")
plt.ylabel("Mental Health Score")
plt.ylim(0, max(mental_health) * 1.15)

plt.text(
    0,
    mental_health[0] + 0.1,
    f"{mental_health[0]:.2f}",
    ha="center"
)

plt.text(
    0.35,
    mental_health[1] + 0.1,
    f"{mental_health[1]:.2f}",
    ha="center"
)

plt.tight_layout()

plt.savefig("figure4_mental_health.png")

plt.show()

# Features and target
features = [
    "stress_level",
    "mental_health_score",
    "sleep_quality",
    "social_support",
    "self_esteem"
]

X = df[features]
y = df["low_performance"]

print("Features:")
print(features)

print("\nTarget distribution:")
print(y.value_counts())

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

# Model pipelines
knn_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=7))
])

logistic_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

print("\nKNN and Logistic Regression pipelines created.")

# Cross validation
knn_cv_scores = cross_val_score(
    knn_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="balanced_accuracy"
)

logistic_cv_scores = cross_val_score(
    logistic_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="balanced_accuracy"
)

print("\nKNN balanced accuracy:")
print(knn_cv_scores)

print("KNN mean:", knn_cv_scores.mean())

print("\nLogistic Regression balanced accuracy:")
print(logistic_cv_scores)

print("Logistic Regression mean:", logistic_cv_scores.mean())

# Train models
knn_pipeline.fit(X_train, y_train)
logistic_pipeline.fit(X_train, y_train)

# Predictions
knn_predictions = knn_pipeline.predict(X_test)
logistic_predictions = logistic_pipeline.predict(X_test)

# Baseline model
baseline = DummyClassifier(
    strategy="most_frequent",
    random_state=RANDOM_STATE
)

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nModels trained successfully.")

# Model performance
results = pd.DataFrame({
    "Model": [
        "Baseline",
        "Logistic Regression",
        "KNN"
    ],
    "Accuracy": [
        accuracy_score(y_test, baseline_predictions),
        accuracy_score(y_test, logistic_predictions),
        accuracy_score(y_test, knn_predictions)
    ],
    "Precision": [
        precision_score(y_test, baseline_predictions),
        precision_score(y_test, logistic_predictions),
        precision_score(y_test, knn_predictions)
    ],
    "Recall": [
        recall_score(y_test, baseline_predictions),
        recall_score(y_test, logistic_predictions),
        recall_score(y_test, knn_predictions)
    ],
    "F1 Score": [
        f1_score(y_test, baseline_predictions),
        f1_score(y_test, logistic_predictions),
        f1_score(y_test, knn_predictions)
    ]
})

print("\nModel Performance Results:")
print(results)

# Classification reports
print("\nBaseline Classification Report")

print(
    classification_report(
        y_test,
        baseline_predictions,
        digits=3
    )
)

print("\nLogistic Regression Classification Report")

print(
    classification_report(
        y_test,
        logistic_predictions,
        digits=3
    )
)

print("\nKNN Classification Report")

print(
    classification_report(
        y_test,
        knn_predictions,
        digits=3
    )
)

# Confusion matrices
cm_logistic = confusion_matrix(
    y_test,
    logistic_predictions
)

cm_knn = confusion_matrix(
    y_test,
    knn_predictions
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    ax=axes[0]
)

axes[0].set_title(
    "Logistic Regression Confusion Matrix"
)

axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    ax=axes[1]
)

axes[1].set_title(
    "KNN Confusion Matrix"
)

axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()

plt.savefig("figure5_confusion_matrices.png")

plt.show()

# Figure 7
models = [
    "Baseline",
    "Logistic Regression",
    "KNN"
]

scores = [
    0.340,
    0.772,
    0.771
]

plt.figure(figsize=(7, 4))

bars = plt.bar(
    [0, 0.35, 0.70],
    scores,
    width=0.25
)

bars[1].set_color("coral")

plt.xticks(
    [0, 0.35, 0.70],
    models
)

plt.title("Figure 7: Model Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)

for x, score in zip(
    [0, 0.35, 0.70],
    scores
):
    plt.text(
        x,
        score + 0.02,
        f"{score:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig("figure7_model_comparison.png")

plt.show()
