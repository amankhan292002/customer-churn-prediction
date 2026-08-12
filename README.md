# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn based on customer, service, contract, and billing information.

The project includes an interactive **Streamlit application** with **SHAP-based Explainable AI** to understand the factors influencing each prediction.

---

## Live Demo

 Try the deployed Streamlit application:

 [https://customer-churn-prediction-292002.streamlit.app](https://customer-churn-prediction-292002.streamlit.app/)

---


## Objective

To build a binary classification model that predicts customer churn:

- `1` → Customer is likely to churn
- `0` → Customer is unlikely to churn

---

## Machine Learning Workflow

```text
Data
  ↓
Data Cleaning & EDA
  ↓
Preprocessing & Encoding
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Serialization
  ↓
Streamlit Application
  ↓
Prediction + SHAP Explanation
```

---

## Model

The final model used in the application is:

**Random Forest Classifier**

The model and preprocessing pipeline are saved using Joblib:

```text
final_random_forest.pkl
preprocessor.pkl
```

---

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## Streamlit Application

The application allows users to enter customer information such as:

- Gender
- Senior Citizen
- Partner & Dependents
- Tenure
- Internet Services
- Online Security & Backup
- Contract
- Payment Method
- Monthly Charges
- Total Charges

The application then displays:

- **Churn Prediction**
- **Churn Probability**
- **Top features influencing the prediction**

---

## Explainable AI with SHAP

SHAP is used to explain individual predictions.

It shows which features are influencing the prediction towards or away from churn.

The application displays the top SHAP features along with a visualization for easier interpretation.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Joblib
- SHAP
- Matplotlib
- Streamlit

---

## Project Structure

```text
Customer-Churn-Project/
│
├── app.py
├── final_random_forest.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Customer-Churn-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 💡 Business Use Case

Customer churn prediction can help companies identify customers who may leave and take preventive actions such as:

- Personalized offers
- Better plans
- Customer support
- Retention campaigns

SHAP explanations can help understand **why a particular customer is predicted to churn**.

---

## Future Improvements

- Deploy the application online
- Compare multiple ML models
- Add customer retention recommendations
- Improve the dashboard and visualizations

---

## Author

**Aman Khan**

Data Science & Generative AI Learner

---

## 🖥️ Application Preview

The Streamlit application provides an interactive interface for customer churn prediction and SHAP-based explanation.

![Customer Churn Prediction App](![alt text](image.png))

---

## Project Highlights

**Machine Learning + Streamlit + Explainable AI**

An end-to-end project covering model training, preprocessing, prediction, deployment, and SHAP-based explanation.


