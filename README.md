# Student Academic Performance Prediction

## Project Overview
This project analyzes student-related factors and uses machine learning classification models to predict whether a student belongs to a low-performance or higher-performance group.
The project focuses on identifying factors related to student academic performance and evaluating machine learning models for classification.

## Dataset

The project uses the Student Stress Level Dataset.
The dataset contains student-related academic, psychological, lifestyle, and social factors.
The main target used in this project is:

- `low_performance` — indicates whether a student's academic performance falls into the low-performance category.

The dataset used for this project is included in the repository as:

`StressLevelDataset.csv`

## Main Features
The analysis uses factors including:
- Stress level
- Depression
- Anxiety
- Sleep quality
- Social support
- Self-esteem
- Academic performance
- Other relevant student factors available in the dataset

A derived mental-health-related score is also used during the analysis.

## Methodology
The project follows these main steps:
1. Load and inspect the dataset.
2. Perform data cleaning and preparation.
3. Create the target classification variable.
4. Split the data into training and testing sets.
5. Establish a majority-class baseline.
6. Train machine learning classification models.
7. Use preprocessing pipelines where required.
8. Compare model performance using cross-validation.
9. Evaluate the final models on the test set.
10. Generate confusion matrices and model comparison visualizations.

## Machine Learning Models
The project evaluates:
- Majority-class baseline
- Logistic Regression
- K-Nearest Neighbors (KNN)

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The project also includes class-wise performance reports and a comparison of the candidate models.

##Repository Structure

student-academic-performance-prediction/
│
├── student_performance.py
├── StressLevelDataset.csv
├── data_dictionary.md
├── final_project_report.docx
├── README.md
├── requirements.txt
│
└── figures/
    ├── graph1.png
    ├── graph2.png
    ├── graph3.png
    ├── graph4.png
    ├── knn_confusion_matrix.png
    ├── logistic_confusion_matrix.png
    └── model_comparison.png
