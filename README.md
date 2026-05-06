# BME 516/616 — Coursework

MATLAB assignments from BME 516/616 (Biomedical Machine Learning). Each script is self-contained and reproducible (`rng(1)` set at the top).

## Contents

| File | Topic |
|---|---|
| `assignment3_diabetes_supervised.m` | Supervised classification of diabetes — Linear SVM, nonlinear SVM (RBF / polynomial), two decision trees (tuned on `MaxNumSplits` vs. `MinLeafSize`), and Naive Bayes. 5-fold CV with Bayesian hyperparameter optimization, full metric table, and a 3×2 grid of confusion matrices. |
| `assignment4_pca_kmeans.m` | Dimensionality reduction and clustering on a gene-expression dataset — PCA with scree and cumulative-variance plots, then k-means for k = 1..10 evaluated by silhouette score and Calinski-Harabasz index. |
| `assignment5_ann_diabetes.m` | Feedforward ANNs for diabetes prediction — `ANN_1` (low LR, underfits), `ANN_2` (LR raised to 1e-3), `ANN_3` (`fitcnet` with full hyperparameter search), and an extra-credit model with batch norm, dropout, and early stopping. ROC curves and a neuron-activation extraction for R2024b. |
| `ehr_missing_outliers.m` | EHR data quality workflow — visualize missingness, compare mean vs. median imputation, detect outliers via ±3 SD, and report before/after summary statistics. |

## Datasets

Datasets are not redistributed here. The scripts expect:
- `Patient_Data_Diabetes.csv` — Pima Indians diabetes data
- `Preprocessed_Diabetes_Dataset.mat`
- `Preprocessed_Gene_Dataset.mat`
- `EHRs_1.xlsx`

## Requirements

- MATLAB R2024b
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox (for `trainnet` / `fitcnet` in the ANN script)

## How to run

Open any script in MATLAB, place the corresponding dataset in the working directory, and run. Each one prints a results summary and produces its figures.
