# Bank-Marketing-Project
## 📌 Project Overview

This project uses Machine Learning to predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign.

The project includes data exploration, data preprocessing, exploratory data analysis (EDA), model training, model evaluation, feature importance analysis, and a Streamlit web application for making predictions.

The trained Machine Learning model is saved using Joblib and integrated into the Streamlit application.

---

## 🎯 Problem Definition

The main objective of this project is to predict the outcome of a bank marketing campaign.

The target variable is:

- `yes` → The customer subscribed to a term deposit.
- `no` → The customer did not subscribe to a term deposit.

This is a **Supervised Machine Learning Classification** problem.

---

## 📊 Dataset

The project uses the **Bank Marketing Dataset**.

The dataset contains information about customers, previous marketing campaigns, and economic indicators.

### Main Feature Categories

#### Customer Information
- Age
- Job
- Marital Status
- Education
- Credit in Default
- Housing Loan
- Personal Loan

#### Campaign Information
- Contact Method
- Contact Month
- Day of Week
- Call Duration
- Number of Contacts
- Previous Contacts
- Previous Campaign Outcome

#### Economic Information
- Employment Variation Rate
- Consumer Price Index
- Consumer Confidence Index
- Euribor 3 Month Rate
- Number of Employees

### Target
- `y`

---

## 🔍 Exploratory Data Analysis

The notebook includes several analysis and visualization steps, including:

- Dataset structure and information
- Statistical summary
- Missing-value analysis
- Duplicate-value analysis
- Target distribution
- Age distribution
- Job distribution
- Education distribution
- Correlation analysis
- Relationships between customer features and subscription

The EDA helps identify patterns and understand the characteristics of customers who subscribe to term deposits.

---

## ⚙️ Data Preprocessing

The following preprocessing techniques are used:

### Numerical Features

Numerical features are standardized using:

`StandardScaler`

### Categorical Features

Categorical features are converted into numerical form using:

`OneHotEncoder`

The encoder uses:

`handle_unknown="ignore"`

This allows the model to handle categorical values that were not present during training.

The preprocessing and Machine Learning model are combined into a single Scikit-learn Pipeline.

---

## 🤖 Machine Learning Model

The project uses:

### Logistic Regression

Logistic Regression is a supervised classification algorithm used to predict whether a customer will subscribe to a term deposit.

The model is trained using:

- 80% Training Data
- 20% Testing Data
- `random_state = 42`
- Stratified splitting

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The exact evaluation results are generated in the notebook and saved in:

`bank_results.pkl`

This ensures that the Streamlit application displays the same results as the notebook.

---

## ⭐ Feature Importance

Feature importance is analyzed using the coefficients of the Logistic Regression model.

The absolute values of the coefficients are used to identify the features that have the strongest influence on the model's predictions.

The project displays the Top 15 important features.

---

## 🌐 Streamlit Web Application

A professional Streamlit dashboard was created to deploy the trained Machine Learning model.

The application contains the following sections:

### 🏠 Overview
Displays:
- Dataset statistics
- Number of customers
- Subscription statistics
- Average age
- Model performance

### 🔮 Make Prediction
Allows the user to enter customer information and receive:

- Subscription prediction
- Subscription probability
- No-subscription probability

### 💬 AI Marketing Assistant

The GUI also contains an AI Marketing Assistant.

Users can ask questions about:

The Machine Learning model
Model performance
Accuracy
Precision
Recall
F1 Score
Important features
Marketing insights
The project methodology

The AI Assistant uses the actual saved project results instead of inventing model results.

### 📊 Data Analysis
Displays:
- Dataset statistics
- Subscription distribution
- Age distribution
- Job analysis
- Dataset sample

### 🤖 Model Performance
Displays:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### ⭐ Feature Importance
Displays:
- Top 15 important features
- Feature importance values
- Logistic Regression coefficients

### ℹ️ About Project
Provides information about:
- Problem definition
- Dataset
- Machine Learning model
- Preprocessing
- Evaluation
- Deployment

---

## 📁 Project Structure

```text
BankMarketingProject/
│
├── app.py
├── bank_model.pkl
├── bank_results.pkl
├── bank-additional-full.csv
└── README.md

🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Streamlit
Jupyter Notebook / Google Colab


📦 Installation

Install the required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
▶️ How to Run the Project
1. Clone or download the project

Make sure all project files are inside the same folder.

2. Open the project folder in Visual Studio Code or another IDE.
3. Install the required libraries
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
4. Run the Streamlit application

Open the terminal in the project folder and run:

streamlit run app.py
5. Open the application

Streamlit will provide a local address in the terminal.

Open it in your web browser to use the application.

🔄 Machine Learning Workflow
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Preprocessing
   ↓
StandardScaler + OneHotEncoder
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Save Model
   ↓
Streamlit Deployment


💡 Project Objective

The system demonstrates how Machine Learning can be used to support bank marketing campaigns by predicting the likelihood of a customer subscribing to a term deposit.

The Streamlit application provides an interactive interface where users can enter customer information and receive a prediction from the trained Machine Learning model.


📌 Conclusion

This project demonstrates a complete Machine Learning workflow, starting from data exploration and preprocessing and ending with model deployment through a Streamlit web application.

The final system allows users to analyze the Bank Marketing dataset, evaluate the trained Logistic Regression model, examine feature importance, and make predictions for individual customers.
