#  Breast Cancer Tumor Classifier using Machine Learning

This project presents a machine learning-based pipeline to classify breast tumors as **benign** or **malignant** using key diagnostic features derived from the [Wisconsin Breast Cancer Dataset (WBCD)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). It demonstrates how AI can aid in early and accurate cancer diagnosis, ultimately contributing to improved clinical decision-making.

---

##  Project Structure

Breast_Cancer_Tumor_Classifier/
│
├── data
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── models/ # Saved models 
├── outputs/ # Plots, confusion matrix, classification report
├── .gitignore
├── requirements.txt
├── README.md
└── Breast_Cancer_Classifier.ipynb # Main notebook

---

##  Objective

To develop a supervised learning model that can **accurately predict the nature of a breast tumor** (malignant or benign) based on input features obtained from fine needle aspirate (FNA) tests of breast masses.

---

##  Dataset Overview

- **Source:** UCI ML Repository via Kaggle  
- **Instances:** 569  
- **Features:** 30 real-valued features (mean, standard error, worst) for:
  - Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness
  - Compactness
  - Concavity, etc.
- **Target:** `Diagnosis` — `M` (Malignant), `B` (Benign)

---

##  Exploratory Data Analysis (EDA)

- Checked for missing/null values
- Visualized:
  - Target distribution
  - Correlation matrix
  - Feature distributions
  - PCA plots for feature dimensionality
- Standardized the data to improve model performance

---

##  Machine Learning Models Used

| Model                  | Accuracy  |
|------------------------|-----------|
| Logistic Regression    | 96.5%     |
| K-Nearest Neighbors    | 95.6%     |
| Random Forest Classifier | 97.3%   |
| Support Vector Machine | 97.0%     |

 **Random Forest** showed the best performance and was selected as the final model.

---

##  Technologies Used

- Python 3.10
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- JupyterLab / Jupyter Notebook
- Git & GitHub

---

##  Results

- Achieved **>97% accuracy** on test data
- Robust classification across both classes
- High **precision**, **recall**, and **F1-score**
- Visualized confusion matrix and ROC curves

---

