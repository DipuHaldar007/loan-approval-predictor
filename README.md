# 🏦 Loan Approval Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a loan application will be approved or rejected using Machine Learning classification algorithms. It analyzes applicant information such as income, credit score, loan amount, employment status, education level, and other financial attributes to predict loan approval.

The main objective is to perform data preprocessing, explore applicant data, compare different classification models, and identify the best-performing model using evaluation metrics and hyperparameter tuning.

## 🚀 Live Demo

🔗 **https://loan-approval-predictor-143.streamlit.app/**

## 🎯 Objectives

- Perform data cleaning and preprocessing.
- Handle missing values in numerical and categorical features.
- Conduct Exploratory Data Analysis (EDA) using visualizations.
- Identify relationships between applicant features and loan approval.
- Apply feature engineering to improve model performance.
- Train and compare multiple Machine Learning classification algorithms.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Use cross-validation and hyperparameter tuning to optimize the Random Forest classifier.

## 🛠️ Technologies Used

- **Python** – Programming language
- **NumPy** – Numerical operations
- **Pandas** – Data manipulation and analysis
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine Learning algorithms and evaluation
- **Jupyter Notebook** – Development environment
- **Joblib** – Saving the trained Machine Learning model

## 🖥️ Streamlit Web Application

An interactive web interface has been developed using **Streamlit** to make loan approval predictions easier to access. Users can enter applicant details through the interface and obtain predictions from the trained Machine Learning model.

### Features
- Interactive interface for entering applicant details.
- User-friendly input form.
- Loan approval prediction using the trained model.
- Easy access to the model's predictions without writing Python code.

## 🤖 Machine Learning Algorithms

The following classification algorithms are implemented and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Random Forest Classifier

Random Forest achieved the strongest overall performance among the initial model comparisons in this notebook.

## 🔄 Project Workflow

1. **Data Loading:** Load the loan approval dataset using Pandas.
2. **Data Preprocessing:** Handle missing values and encode categorical variables.
3. **Exploratory Data Analysis:** Analyze loan approval distribution, applicant income, credit scores, loan amounts, and other features using visualizations.
4. **Feature Engineering:** Create additional features, including squared credit score and debt-to-income ratio features.
5. **Feature Scaling:** Apply StandardScaler to prepare features for suitable classification algorithms.
6. **Model Training:** Train Logistic Regression, KNN, Gaussian Naive Bayes, and Random Forest models.
7. **Model Evaluation:** Compare accuracy, precision, recall, F1-score, and confusion matrices.
8. **Cross-Validation:** Use 5-fold cross-validation to assess model consistency.
9. **Hyperparameter Tuning:** Use GridSearchCV to find suitable Random Forest hyperparameters.
10. **Final Evaluation:** Evaluate the optimized model on the test set using a classification report and confusion matrix.
11. **Model Saving:** Save the optimized model using Joblib.

## 📊 Model Performance

Initial model comparison results from the notebook:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 88.00% |
| KNN | 78.50% |
| Gaussian Naive Bayes | 86.00% |
| Random Forest | 90.50% |

The Random Forest classifier achieved a **mean cross-validation accuracy of 93.62%** in the notebook's 5-fold cross-validation experiment.

These figures represent the recorded experiments before the final hyperparameter-tuned evaluation. Final test performance may differ.

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── data/
│   └── loan_approval_data.csv
│
├── model/
│   └── tuned_loan_model.pkl
│
├── notebook/
│   └── Loan_project.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
└── app.py
```

- `Loan_project.ipynb` – Complete Machine Learning workflow.
- `loan_approval_data.csv` – Dataset used for training and evaluation.
- `tuned_loan_model.pkl` – Saved optimized Random Forest model, generated when the notebook's saving cell is executed.
- `README.md` – Project documentation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the project directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `Loan_project.ipynb` and run the cells in sequence.

**Note:** Keep `loan_approval_data.csv` in the appropriate location expected by the notebook. Without the dataset, the notebook will not run from the beginning.

## 💾 Saving the Trained Model

The optimized Random Forest model is saved using Joblib:

```python
import joblib

joblib.dump(best_rf_model, "tuned_loan_model.pkl")
```

This creates a model file that can be loaded later for predictions, provided the required preprocessing steps and feature order are preserved.

## 🚀 Future Improvements

- Improve model performance using additional experiments and validation techniques.
- Explore other classification algorithms and compare their performance.
- Enhance the Streamlit interface with additional visualizations and prediction insights.
- Deploy and maintain the application online for public demonstration.

## 📚 Learning Outcomes

Through this project, I practiced:

- Data preprocessing and exploratory data analysis.
- Feature engineering and feature scaling.
- Supervised Machine Learning classification.
- Model comparison and evaluation.
- Cross-validation and hyperparameter optimization.
- Saving trained models for future use.

## 👨‍💻 Author

**Deptendra Haldar**

B.Tech CSE (AI & ML) Student

---

⭐ If you find this project useful, consider giving the repository a star!
