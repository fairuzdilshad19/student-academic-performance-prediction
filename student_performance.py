# LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score


# LOAD DATASET
url="https://raw.githubusercontent.com/mibur1/psy111/main/book/statistics/4_Moderated_Reg/data/StressLevelDataset.csv"

df_raw=pd.read_csv(url)
df=df_raw.copy()


# BASIC DATA CHECKING
print("FIRST 10 ROWS")
print(df.head(10))

print("\nLAST 10 ROWS")
print(df.tail(10))

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATA TYPES")
print(df.dtypes)

print("\nDATA INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nDUPLICATES")
print(df.duplicated().sum())


#DATA CLEANING
df=df.drop(columns=["blood_pressure","headache"])

#FILL MISSING NUMERICAL VALUES WITH MEDIAN
for col in df.select_dtypes(include=np.number).columns:
    df[col]=df[col].fillna(df[col].median())

#REMOVE DUPLICATES
df=df.drop_duplicates()

#RESET ROW NUMBERS
df.reset_index(drop=True,inplace=True)

#CREATE MENTAL HEALTH SCORE
df["mental_health_score"]=(
    df["depression"]+df["anxiety_level"]
)/2

print("\nMENTAL HEALTH SCORE")

print(
    df[
        ["depression","anxiety_level","mental_health_score"]
    ].head(10)
)

# CREATE PERFORMANCE CLASS
# 1 = LOW PERFORMANCE
# 0 = HIGH PERFORMANCE

df["low_performance"]=(df["academic_performance"]<=2).astype(int)

# REMOVE ACADEMIC PERFORMANCE
df=df.drop(columns=["academic_performance"])

# FINAL CHECK
print("\nFINAL DATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isna().sum())

print("\nFINAL FIRST 10 ROWS")
print(df.head(10))

print("\nFINAL LAST 10 ROWS")
print(df.tail(10))


# GRAPH 1 - PERFORMANCE DISTRIBUTION
counts=df["low_performance"].value_counts().sort_index()

plt.figure(figsize=(7,4))

values=[
    counts[0],
    counts[1]
]

plt.bar(
    ["High Performance","Low Performance"],
    values,
    width=0.35
)

plt.title("Academic Performance Distribution")
plt.xlabel("Performance Group")
plt.ylabel("Number of Students")

plt.ylim(0,max(values)*1.12)

for i,value in enumerate(values):
    plt.text(
        i,
        value+5,
        str(value),
        ha="center"
    )

plt.tight_layout()
plt.savefig("graph1_performance.png")
plt.show()


# GRAPH 2 - STRESS LEVEL BY PERFORMANCE
stress=df.groupby(
    "low_performance"
)["stress_level"].mean()

plt.figure(figsize=(7,4))

plt.bar(
    [0,0.35],
    [stress[0],stress[1]],
    width=0.25,
    color=["steelblue","lightcoral"]
)

plt.xticks(
    [0,0.35],
    ["High Performance","Low Performance"]
)

plt.title("Average Stress Level by Performance Group")
plt.xlabel("Performance Group")
plt.ylabel("Average Stress Level")

plt.ylim(0,max(stress)*1.15)

plt.text(
    0,
    stress[0]+0.04,
    f"{stress[0]:.2f}",
    ha="center"
)

plt.text(
    0.35,
    stress[1]+0.04,
    f"{stress[1]:.2f}",
    ha="center"
)

plt.tight_layout()
plt.savefig("graph2_stress.png")
plt.show()\

# GRAPH 3 - SLEEP QUALITY BY PERFORMANCE
sleep=df.groupby(
    "low_performance"
)["sleep_quality"].mean()

plt.figure(figsize=(7,4))

plt.bar(
    [0,0.35],
    [sleep[0],sleep[1]],
    width=0.25,
    color=["steelblue","lightcoral"]
)

plt.xticks(
    [0,0.35],
    ["High Performance","Low Performance"]
)

plt.title("Average Sleep Quality by Performance Group")
plt.xlabel("Performance Group")
plt.ylabel("Average Sleep Quality")

plt.ylim(0,max(sleep)*1.15)

plt.text(
    0,
    sleep[0]+0.05,
    f"{sleep[0]:.2f}",
    ha="center"
)

plt.text(
    0.35,
    sleep[1]+0.05,
    f"{sleep[1]:.2f}",
    ha="center"
)

plt.tight_layout()
plt.savefig("graph3_sleep.png")
plt.show()


# GRAPH 4 - MENTAL HEALTH SCORE
mental_health=df.groupby(
    "low_performance"
)["mental_health_score"].mean()

plt.figure(figsize=(7,4))

plt.bar(
    [0,0.35],
    [mental_health[0],mental_health[1]],
    width=0.25,
    color=["steelblue","lightcoral"]
)

plt.xticks(
    [0,0.35],
    ["High Performance","Low Performance"]
)

plt.title("Average Mental Health Score by Performance Group")
plt.xlabel("Performance Group")
plt.ylabel("Mental Health Score")

plt.ylim(0,max(mental_health)*1.15)

plt.text(
    0,
    mental_health[0]+0.1,
    f"{mental_health[0]:.2f}",
    ha="center"
)

plt.text(
    0.35,
    mental_health[1]+0.1,
    f"{mental_health[1]:.2f}",
    ha="center"
)

plt.tight_layout()
plt.savefig("graph4_mental_health.png")
plt.show()


# SELECT FEATURES AND TARGET
features=[
    "stress_level",
    "mental_health_score",
    "sleep_quality",
    "social_support",
    "self_esteem"
]

X=df[features]

y=df["low_performance"]

print("\nFEATURES")
print(X.head(10))

print("\nTARGET")
print(y.head(10))


# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

RANDOM_STATE=42

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTRAINING DATA")
print(X_train.shape)

print("\nTESTING DATA")
print(X_test.shape)


# KNN MODEL
knn=Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scale",StandardScaler()),
    ("model",KNeighborsClassifier(n_neighbors=7))
])

knn.fit(
    X_train,
    y_train
)


# LOGISTIC REGRESSION MODEL
logistic=Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scale",StandardScaler()),
    ("model",LogisticRegression(max_iter=1000))
])

logistic.fit(
    X_train,
    y_train
)


# CROSS VALIDATION
knn_cv=cross_val_score(
    knn,
    X_train,
    y_train,
    cv=5,
    scoring="balanced_accuracy"
)

logistic_cv=cross_val_score(
    logistic,
    X_train,
    y_train,
    cv=5,
    scoring="balanced_accuracy"
)

print(
    "\nKNN CV Balanced Accuracy:",
    knn_cv.mean()
)

print(
    "Logistic Regression CV Balanced Accuracy:",
    logistic_cv.mean()
)


# SELECT MODEL
if knn_cv.mean()>logistic_cv.mean():

    best_model=knn
    best_model_name="KNN"

else:

    best_model=logistic
    best_model_name="Logistic Regression"

print(
    "\nSelected Model:",
    best_model_name
)


#FINAL EVALUATION
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# BASELINE
baseline=DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(
    X_train,
    y_train
)

baseline_pred=baseline.predict(X_test)

baseline_accuracy=accuracy_score(
    y_test,
    baseline_pred
)

baseline_precision=precision_score(
    y_test,
    baseline_pred,
    zero_division=0
)

baseline_recall=recall_score(
    y_test,
    baseline_pred,
    zero_division=0
)

baseline_f1=f1_score(
    y_test,
    baseline_pred,
    zero_division=0
)

print("\nBASELINE")

print("Accuracy:",baseline_accuracy)
print("Precision:",baseline_precision)
print("Recall/Sensitivity:",baseline_recall)
print("F1 Score:",baseline_f1)

# LOGISTIC REGRESSION
logistic_pred=logistic.predict(X_test)

logistic_accuracy=accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision=precision_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_recall=recall_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_f1=f1_score(
    y_test,
    logistic_pred,
    zero_division=0
)

print("\nLOGISTIC REGRESSION")

print("Accuracy:",logistic_accuracy)
print("Precision:",logistic_precision)
print("Recall/Sensitivity:",logistic_recall)
print("F1 Score:",logistic_f1)


print("\nLOGISTIC REGRESSION CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=[
            "High Performance",
            "Low Performance"
        ],
        zero_division=0
    )
)


logistic_confusion_matrix=confusion_matrix(
    y_test,
    logistic_pred
)

print(
    "Logistic Regression Confusion Matrix"
)

print(
    logistic_confusion_matrix
)

# KNN
knn_pred=knn.predict(X_test)

knn_accuracy=accuracy_score(
    y_test,
    knn_pred
)
knn_precision=precision_score(
    y_test,
    knn_pred,
    zero_division=0
)
knn_recall=recall_score(
    y_test,
    knn_pred,
    zero_division=0
)
knn_f1=f1_score(
    y_test,
    knn_pred,
    zero_division=0
)
print("\nKNN")

print("Accuracy:",knn_accuracy)
print("Precision:",knn_precision)
print("Recall/Sensitivity:",knn_recall)
print("F1 Score:",knn_f1)


print("\nKNN CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        knn_pred,
        target_names=[
            "High Performance",
            "Low Performance"
        ],
        zero_division=0
    )
)

knn_confusion_matrix=confusion_matrix(
    y_test,
    knn_pred
)
print(
    "KNN Confusion Matrix"
)
print(
    knn_confusion_matrix
)

# MODEL COMPARISON TABLE
comparison=pd.DataFrame({

    "Model":[
        "Baseline",
        "Logistic Regression",
        "KNN"
    ],

    "Accuracy":[
        baseline_accuracy,
        logistic_accuracy,
        knn_accuracy
    ],

    "Precision":[
        baseline_precision,
        logistic_precision,
        knn_precision
    ],

    "Recall/Sensitivity":[
        baseline_recall,
        logistic_recall,
        knn_recall
    ],

    "F1 Score":[
        baseline_f1,
        logistic_f1,
        knn_f1
    ]

})

print("\nMODEL COMPARISON")

print(comparison)

# BEST MODEL
if logistic_f1>knn_f1:

    best_test_model="Logistic Regression"
    best_f1=logistic_f1
else:
    best_test_model="KNN"
    best_f1=knn_f1

print(
    "\nBEST MODEL:",
    best_test_model
)

print(
    "F1 Score:",
    best_f1
)

# BEST MODEL VS BASELINE
if best_f1>baseline_f1:
    print(
        "The best model performs better than the baseline."
    )
else:
    print(
        "The best model does not perform better than the baseline."
    )

# LOGISTIC REGRESSION CONFUSION MATRIX
plt.figure(figsize=(6,5))

sns.heatmap(
    logistic_confusion_matrix,
    annot=True,
    fmt="d",
    cmap="GnBu",
    xticklabels=["High","Low"],
    yticklabels=["High","Low"]
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("logistic_confusion_matrix.png")
plt.show()

# KNN CONFUSION MATRIX
plt.figure(figsize=(6,5))

sns.heatmap(
    knn_confusion_matrix,
    annot=True,
    fmt="d",
    cmap="GnBu",
    xticklabels=["High","Low"],
    yticklabels=["High","Low"]
)

plt.title(
    "KNN Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("knn_confusion_matrix.png")
plt.show()

# FINAL MODEL COMPARISON GRAPH
models=[
    "Baseline",
    "Logistic Regression",
    "KNN"
]
scores=[
    baseline_f1,
    logistic_f1,
    knn_f1
]

# HIGHEST F1 SCORE = CORAL
if baseline_f1==max(scores):
    colors=[
        "coral",
        "steelblue",
        "steelblue"
    ]

elif logistic_f1==max(scores):
    colors=[
        "steelblue",
        "coral",
        "steelblue"
    ]

else:
    colors=[
        "steelblue",
        "steelblue",
        "coral"
    ]

plt.figure(figsize=(7,4))

plt.bar(
    [0,0.35,0.70],
    scores,
    width=0.25,
    color=colors
)

plt.xticks(
    [0,0.35,0.70],
    models
)
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")

plt.ylim(0,1)

plt.text(
    0,
    scores[0]+0.02,
    f"{scores[0]:.3f}",
    ha="center"
)
plt.text(
    0.35,
    scores[1]+0.02,
    f"{scores[1]:.3f}",
    ha="center"
)
plt.text(
    0.70,
    scores[2]+0.02,
    f"{scores[2]:.3f}",
    ha="center"
)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
