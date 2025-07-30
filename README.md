# 🧬 Breast Cancer Tumor Classifier

This project aims to build a machine learning-based classifier to distinguish between **benign** and **malignant** breast cancer tumors using features derived from digitized histopathology images (Wisconsin Breast Cancer Dataset). The classifier is deployed as a **Streamlit web application**.

## 🚀 Live Demo
Try the app here: [Breast Cancer Tumor Classifier](https://breastcancertumorclassifier-qezewv5pqknpw4fitspg5k.streamlit.app/)
e

---

## 📁 Project Structure

Breast_Cancer_Tumor_Classifier/
├── app.py # Streamlit web app
├── model.pkl # Trained ML model
├── data/
│ └── breast_cancer_data.csv
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_model_evaluation.ipynb
├── README.md
├── requirements.txt
└── .gitignore

---

## 📊 Dataset Used

- **Source**: UCI Machine Learning Repository – [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Samples**: 569
- **Features**: 30 real-valued input features (mean, standard error, and worst values of various cell nuclei characteristics)
- **Label**: Diagnosis (`M` = Malignant, `B` = Benign)

---

## 🔍 Project Workflow

1. **Data Exploration**
   - Checked for null values, duplicate entries
   - Visualized class distribution and feature correlations

2. **Preprocessing**
   - Label encoding (`M` → 1, `B` → 0)
   - Feature scaling using `StandardScaler`

3. **Model Training**
   - Trained a **Random Forest Classifier** on the dataset
   - Performed train-test split (80:20)
   - Saved model using `pickle` as `model.pkl`

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score calculated
   - Confusion matrix plotted

5. **Deployment**
   - Built a simple **Streamlit app** that accepts key features and returns prediction in real time
   - Hosted via [Streamlit Community Cloud](https://streamlit.io/cloud)
   - live demo [Breast Cancer Tumor Classifier](https://breastcancertumorclassifier-qezewv5pqknpw4fitspg5k.streamlit.app/)

---

## ✅ Results

| Metric        | Value   |
|---------------|---------|
| Accuracy       | 96.49%  |
| Precision      | 95.83%  |
| Recall         | 96.15%  |
| F1-score       | 95.99%  |

> ✔️ Results are based on evaluation of the Random Forest model using test data split (20%) from the dataset.

---

## 💻 Requirements

To run this project locally, install the following Python packages:

```bash
pip install -r requirements.txt
numpy
pandas
scikit-learn
streamlit
matplotlib
seaborn

## 🛠️ Technologies Used
Category	Tools & Libraries
Language	Python 3.11
IDE	JupyterLab, VS Code
Libraries	pandas, numpy, scikit-learn, matplotlib, seaborn
Web App	Streamlit
Version Control	Git & GitHub
