import streamlit as st
import pandas as pd
import joblib

st.title("💳 Credit Card Fraud Detection")

# Load model
model = joblib.load("xgb_model.pkl")  # Ensure this file is in the same directory

# Upload file
uploaded_file = st.file_uploader("Upload a test_input.csv", type=["csv"])
if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)

    # Ensure all columns are numeric
    data = data.astype(float)

    # Predict
    prediction = model.predict(data)
    proba = model.predict_proba(data)[:, 1]  # Fraud probability

    # Show results
    result_df = data.copy()
    result_df["Prediction"] = prediction
    result_df["Fraud_Probability (%)"] = (proba * 100).round(2)

    st.success("✅ Prediction completed.")
    st.write(result_df)

    # Optional: Filter fraudulent predictions
    frauds = result_df[result_df["Prediction"] == 1]
    if not frauds.empty:
        st.warning(f"⚠️ {len(frauds)} potential fraudulent transaction(s) detected!")
        st.dataframe(frauds)
    else:
        st.info("✅ No fraud detected in uploaded transactions.")











# Save this as app.py

# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load("xgb_model.pkl")

# st.title("💳 Credit Card Fraud Detection")

# uploaded_file = st.file_uploader("Upload CSV File for Prediction", type=["csv"])

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     prediction = model.predict(data)
#     st.write("🔍 Predictions:")
#     st.write(prediction)
