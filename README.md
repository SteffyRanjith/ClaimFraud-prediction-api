# 🛡️ Health Insurance Fraud Detection API

This project is a **machine learning–powered fraud detection system** for health insurance claims.  
It analyzes claim details such as patient info, provider details, billed amounts, and procedure codes to determine the likelihood of fraudulent activity.

## 📌 Project Overview
Insurance fraud is a major challenge for the healthcare industry, leading to financial losses and compromised patient care.  
This system uses a **Random Forest Classifier** to detect suspicious claims by analyzing historical data patterns.

## 🛠 Tech Stack
- **Python 3.9+**
- **FastAPI** – For serving the prediction API
- **scikit-learn** – Model training & evaluation
- **Pandas / NumPy** – Data preprocessing
- **ngrok** – Public URL exposure for local API

## 🚀 Steps Followed
1. **Data Preprocessing** – Removed irrelevant identifiers, handled missing values, and engineered new features like billing differences and submission delays.  
2. **Encoding & Scaling** – Converted categorical variables using **Label Encoding** and normalized numeric features with **StandardScaler**.  
3. **Model Training** – Trained a **Random Forest** model with **Leave-One-Out Cross-Validation** for robust evaluation.  
4. **API Development** – Built a **FastAPI** backend to expose a prediction endpoint for real-time claim assessment.

