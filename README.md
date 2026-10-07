# Diabetes Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a patient is likely to have diabetes using machine learning techniques.

The project covers the complete machine learning workflow, including data cleaning, exploratory data analysis (EDA), preprocessing, model comparison, hyperparameter tuning, class balancing, threshold tuning, and model evaluation.

A **Balanced Random Forest** model was selected as the final model with a classification threshold of **0.35**, prioritizing the detection of diabetic cases and reducing false negatives.

---

## 🎯 Objective

The main objective of this project is to build a classification model that can effectively identify potential diabetic cases while paying particular attention to reducing **false negatives**.

---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains medical diagnostic measurements such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome

`Outcome` is the target variable:

- `0` → Non-Diabetic
- `1` → Diabetic

The dataset contains **768 patient records** and **8 input features**.

---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

- Dataset structure and statistical summary
- Missing-value analysis
- Duplicate-value analysis
- Zero-value analysis
- Target class distribution
- Histograms
- Correlation heatmap
- Outlier analysis using boxplots
- Feature relationship analysis with respect to the target

### Important Data Cleaning

Zero values in the following medically relevant features were treated as missing values:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

These zero values were replaced with `NaN` and handled using **median imputation fitted only on the training data** to avoid data leakage.

Outliers were retained because they may represent genuine medical observations.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

1. Train-test split using stratification
2. Median imputation for missing values
3. Standardization using `StandardScaler`
4. Feature-target separation

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

---

## 🤖 Models Implemented

The following classification algorithms were evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

Random Forest was then further optimized using **GridSearchCV** with 5-fold cross-validation.

---

## 🔧 Random Forest Hyperparameter Tuning

GridSearchCV was used to tune parameters including:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

The optimization metric was **F1 Score**.

The tuned Random Forest achieved:

- Accuracy: **74.03%**
- Precision: **65.91%**
- Recall: **53.70%**
- F1 Score: **59.18%**
- ROC-AUC: **\~0.81**

Although the model provided reasonable overall performance, its relatively low recall indicated that a considerable number of diabetic cases were being missed.

---

## ⚖️ Class Balancing

The dataset contains approximately:

- **65% Non-Diabetic**
- **35% Diabetic**

To give greater importance to the minority class, a **Balanced Random Forest using `class_weight='balanced'`** was implemented.

The balanced model was also optimized using GridSearchCV.

---

## 🎚️ Threshold Tuning

Instead of relying only on the default classification threshold of `0.50`, different thresholds were evaluated:

- 0.30
- 0.35
- 0.40
- 0.45
- 0.50

A threshold of **0.35** provided the best balance between Recall and F1 Score for the diabetic class.

### Threshold = 0.35

- Accuracy: **76.0%**
- Precision: **61.0%**
- Recall: **87.0%**
- F1 Score: **71.8%**

The threshold was selected because the project prioritizes identifying diabetic cases and minimizing false negatives.

---

## 📈 Confusion Matrix

For the final Balanced Random Forest at a threshold of 0.35:

```text
[[70, 30],
 [ 7, 47]]
```

This corresponds to:

- **True Negatives:** 70
- **False Positives:** 30
- **False Negatives:** 7
- **True Positives:** 47

Compared with the original Random Forest, false negatives were reduced from **25 to 7**, while true positives increased from **29 to 47**.

This demonstrates the benefit of prioritizing diabetic-case detection.

---

## 📋 Final Classification Report

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Non-Diabetic (0) | 0.91      | 0.70   | 0.79     |
| Diabetic (1)     | 0.61      | 0.87   | 0.72     |

**Overall Accuracy: 76%**

---

## ⭐ Feature Importance

The Random Forest feature importance analysis showed that:

1. **Glucose** was the most influential feature
2. **BMI** was among the important features
3. **Age** was also an important predictor

This indicates that these features contributed substantially to the model's predictions.

---

## 💾 Model Saving

The final model was saved using `joblib` as a model bundle containing:

- Trained Balanced Random Forest
- Median imputer
- Standard scaler
- Feature names
- Selected classification threshold

This allows the complete preprocessing and prediction pipeline to be reused in the application.

---

## 🌐 Streamlit Application

An interactive Streamlit application is being developed for this project.

The application will allow users to enter patient-related feature values and receive a model-based diabetes prediction.

> **Note:** This project is intended for educational and machine learning demonstration purposes and should not be used as a substitute for professional medical diagnosis.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 📁 Project Structure

```text
Diabetes-Prediction/
│
├── Data/
│   └── diabetes.csv
│
├── Model/
│   └── diabetes_model_bundle.joblib
│
├── Notebook/
│   └── Diabetes_Prediction.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Key Takeaways

- Performed complete EDA and data preprocessing.
- Identified and handled medically implausible zero values.
- Avoided data leakage during imputation.
- Compared multiple classification algorithms.
- Performed Random Forest hyperparameter tuning.
- Addressed class imbalance using `class_weight='balanced'`.
- Performed classification threshold tuning.
- Reduced diabetic false negatives from **25 to 7**.
- Achieved **87% recall** for the diabetic class.
- Saved the complete model pipeline using Joblib.
- Built an interactive Streamlit application for deployment.

---

## 👨‍💻 Author

**Shubham Mathur**

Machine Learning / Data Science Enthusiast

GitHub: [https://github.com/shubhammathur9863]

LinkedIn: [https://www.linkedin.com/in/shubham-mathur03/?isSelfProfile=true]