# 🕵️‍♂️ HR Employee Attrition Prediction & Analysis

## 📌 Project Overview
Employee attrition is a critical challenge for organizations, leading to high replacement costs and loss of valuable talent. This project aims to build a robust Machine Learning pipeline to predict which employees are at risk of leaving, allowing the HR department to take proactive retention measures.

## 🚀 Business Value & Strategy
Instead of optimizing for overall accuracy (which is misleading in highly imbalanced HR datasets), this project prioritizes **Recall**. 
By implementing a **Linear SVM** paired with **SMOTEENN**, the final model successfully identifies **~81% of at-risk employees**, providing HR with a highly actionable "watch list" to conduct stay-interviews before resignation letters are drafted.

## 🛠️ Tech Stack & Methodologies
* **Language:** Python
* **Libraries:** Pandas, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn
* **Data Preprocessing:** Custom Pipelines, RobustScaler, OneHotEncoding.
* **Handling Imbalanced Data:** `SMOTEENN` (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbours).
* **Modeling:** Logistic Regression, Random Forest, KNN, and Support Vector Machines (SVM).
* **Hyperparameter Tuning:** `GridSearchCV` with Stratified K-Fold Cross Validation.

## 📊 Key Findings (Feature Importance)
By extracting coefficients from the Tuned Linear SVM, we identified the top drivers of employee turnover:
1. **Job Role (Laboratory Technician):** Highest risk factor, indicating potential issues with compensation or role design.
2. **OverTime:** Both excessive overtime and strict "no overtime" policies correlate with higher attrition (burnout vs. disengagement).
3. **Business Travel:** Frequent travelers and non-travelers leave more often than those who travel moderately.
4. **Role Stagnation (`YearsInCurrentRole`):** Lack of internal mobility significantly drives talent away.
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/9109e5f9-4d80-4bea-b81f-16a07dd7b305" />

## 📈 Model Performance
* **Algorithm:** Linear Support Vector Machine (SVM)
* **Threshold:** Default (0.50) - chosen for best practical HR balance.
* **Recall (Class 1):** `0.809` (Caught 29 out of 47 leaving employees)
* **ROC AUC:** `0.830`

## 📂 Repository Structure
* `V11_Employee_Attrition_Performance_prediction.ipynb` : The complete End-to-End code (EDA, Pipeline, Modeling, Evaluation).
* `WA_Fn-UseC_-HR-Employee-Attrition.csv` : The HR dataset used for training and testing. [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## 💡 How to Run
1. Clone the repository: `git clone https://github.com/omarDlgaber/HR-Attrition-Prediction.git`
2. Install the required dependencies.
3. Run the Jupyter Notebook.
