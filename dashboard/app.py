import streamlit as st
import pickle
import pandas as pd
import xgboost


# 📥 Load the saved XGBoost model
with open('models/fraud_detection_xgboost_v1.pkl', 'rb') as file:
    model = pickle.load(file)

# 🎨 Streamlit Page Config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

# 🚀 Title
st.title("💳 Credit Card Fraud Detection")

st.write(
    """
    This application uses a machine learning model to detect fraudulent credit card transactions.
    Fill the details below or upload a CSV file to make predictions.
    """
)

# 🛠 Function to make prediction
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# 📥 Sidebar: Option to Upload CSV or Manual Input
st.sidebar.header("Input Options")

input_choice = st.sidebar.radio("Choose input method:", ('Manual Input', 'Upload CSV'))

if input_choice == 'Manual Input':
    st.header("Enter Transaction Details Manually:")

    # Manual Inputs
    scaled_amount = st.number_input('Scaled Amount', min_value=-10.0, max_value=10.0, value=0.0)
    scaled_time = st.number_input('Scaled Time', min_value=-10.0, max_value=10.0, value=0.0)

    features = []
    for i in range(1, 29):
        val = st.number_input(f'V{i}', min_value=-30.0, max_value=30.0, value=0.0)
        features.append(val)

    # Corrected Collecting all inputs
    input_features = [scaled_amount, scaled_time] + features

    # Correct feature names
    feature_names = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

    # Create DataFrame with correct feature names
    input_df = pd.DataFrame([input_features], columns=feature_names)

    if st.button("Predict"):
        prediction = predict(input_df)
        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction.")

else:
    st.header("Upload CSV File:")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", input_df.head())

        if st.button("Predict on Uploaded Data"):
            prediction = predict(input_df)
            input_df['Prediction'] = prediction
            st.write(input_df)

            # Show number of fraudulent transactions detected
            fraud_count = sum(prediction)
            st.warning(f"🚨 Detected {fraud_count} fraudulent transactions out of {len(prediction)}.")

# Footer
st.markdown("---")
st.markdown("Made for GrowthLink Internship Project")