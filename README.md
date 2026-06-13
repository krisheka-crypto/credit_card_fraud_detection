# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project uses Machine Learning techniques to detect fraudulent credit card transactions. Due to the highly imbalanced nature of fraud datasets, multiple classification models were trained and evaluated to identify the most effective approach.

The project also includes a Streamlit web application that allows users to upload transaction data and perform fraud detection in real time.

---

## Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Fraud vs Non-Fraud Transaction Analysis
* Multiple Machine Learning Models Comparison
* Performance Evaluation using Precision, Recall, and F1 Score
* Interactive Streamlit Web Application
* CSV Upload and Prediction Functionality

---

## Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized transaction features.

**Note:** The dataset is not included in this repository due to GitHub file size limitations.

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Machine Learning Models Evaluated

### Logistic Regression

* Precision: 0.8659
* Recall: 0.7245
* F1 Score: 0.7889

### Decision Tree

* Precision: 0.7526
* Recall: 0.7449
* F1 Score: 0.7487

### Gradient Boosting

* Precision: 0.5294
* Recall: 0.1837
* F1 Score: 0.2727

### Random Forest (Best Model)

* Precision: 0.9412
* Recall: 0.8163
* F1 Score: 0.8743

---

## Project Structure

credit_card_fraud_detection/

├── app.py

├── fraud_detection.py

├── fraud_detection_model.pkl

├── .gitignore

└── README.md

---

## How to Run

### Clone the Repository

git clone https://github.com/krisheka-crypto/credit_card_fraud_detection.git

cd credit_card_fraud_detection

### Install Dependencies

pip install -r requirements.txt

### Run the Application

streamlit run app.py

---

## Results

The Random Forest Classifier achieved the best overall performance with:

* Precision: 94.12%
* Recall: 81.63%
* F1 Score: 87.43%

These results demonstrate strong effectiveness in identifying fraudulent transactions while minimizing false positives.

---

<img width="1600" height="848" alt="image" src="https://github.com/user-attachments/assets/bb71336c-d46b-4dd9-8df7-bf5a4b3946cb" />

<img width="1600" height="850" alt="image" src="https://github.com/user-attachments/assets/f408f09e-09d2-4d47-8070-7b79e7b4a0e0" />

<img width="1600" height="851" alt="image" src="https://github.com/user-attachments/assets/560243d6-e023-4db9-b62c-118034033a7e" />

<img width="1600" height="876" alt="image" src="https://github.com/user-attachments/assets/290d4162-7edf-4a99-a512-bae14fb88d93" />

<img width="1600" height="876" alt="image" src="https://github.com/user-attachments/assets/ee0e451f-a98c-4d2d-9c85-477a49423da4" />

## Future Improvements

* Hyperparameter Tuning
* Advanced Ensemble Models
* Real-Time Fraud Detection Pipeline
* Model Deployment on Cloud Platforms
* Explainable AI (XAI) Integration

---

## Author

Krisheka

B.E. Electronics and Communication Engineering (ECE)

Aspiring AI/ML Engineer
