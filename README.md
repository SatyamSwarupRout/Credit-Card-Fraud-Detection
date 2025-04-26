# **💳 Credit Card Fraud Detection 🚨**

A machine learning project to detect fraudulent credit card transactions using classification models and Streamlit dashboard.

# **🚀 **Live Dashboard**:

Live Link: https://credit-card-fraud-detection-gv2aqma7q9ypgvrmspytb4.streamlit.app

### **📂 Project Structure**

Credit_Card_Fraud_Detection/

│

├── models/

│   └── fraud_detection_xgboost_v1.pkl

│

├── notebooks/

│   └── fraud_detection.ipynb

│

├── dashboard/

│   └── app.py

│
├── data/

│   └── (dataset files if needed)

│

├── README.md

└── requirements.txt


### **📚 Problem Statement**

Credit card fraud is a serious problem causing billions of dollars of losses every year.
The objective is to build an accurate fraud detection model that minimizes false positives and predicts fraudulent transactions effectively.

### **📊 Dataset Used**

Source: Kaggle - Credit Card Fraud Detection (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Contains anonymized features (V1–V28) obtained via PCA transformation.

Highly imbalanced dataset (fraudulent transactions are very rare).


### **⚙️ Techniques Used**

**Data Preprocessing (Scaling, Feature Engineering)**

**Handling Class Imbalance using SMOTE**

 **Model Training:**

XGBoost Classifier ✅

Random Forest Classifier

Logistic Regression

LightGBM

CatBoost

Extra Trees

**Hyperparameter Tuning**

**Evaluation using:**

Confusion Matrix

Classification Report

ROC-AUC Score

### **🖥️ Streamlit Dashboard**

Built a simple and interactive Streamlit app to:

Enter transaction details manually 🔢

Upload CSV files and predict fraud on batch data 📂

See prediction results directly on the dashboard 📈

### **🛠 How to Run Locally**
**1. Clone the repository**

git clone https://github.com/SatyamSwarupRout/Credit-Card-Fraud-Detection.git

cd Credit_Card_Fraud_Detection

**2. Create and activate virtual environment**

conda create -n fraud_env python=3.10

conda activate fraud_env

**3. Install required libraries**

pip install -r requirements.txt

**4. Run the Streamlit app**

streamlit run dashboard/app.py

✅ Open browser ➔ http://localhost:8501/

### **📈 Results Achieved**

**Model-----------------------------------ROC-AUC Score-------------------Accuracy-----------------------Special Notes**

**XGBoost**-------------------------------0.90+----------------------------------99.92%---------------------------Best balance between fraud detection and false positives

**Random Forest**----------------------0.89+----------------------------------99.92%---------------------------Good

**Logistic Regression**---------------0.92+----------------------------------98.00%---------------------------Very high false positive

**LightGBM**------------------------------0.91+---------------------------------99.90%---------------------------Good

**CatBoost**-------------------------------0.91+---------------------------------99.90%---------------------------Good

**Extra Trees**----------------------------0.89+---------------------------------99.91%---------------------------Good

✅ XGBoost selected as the final model!

### **📦 Future Improvements**

Fine-tuning hyperparameters further

Real-time streaming fraud detection

Deployment on cloud platforms (AWS / Heroku)

### **✨ Acknowledgements**
Dataset provided by Kaggle: Credit Card Fraud Detection

Streamlit for easy dashboard building

XGBoost, Scikit-learn, Pandas, Matplotlib

**🚀 Made for GrowthLink Internship Project**
