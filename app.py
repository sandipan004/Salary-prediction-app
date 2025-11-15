import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model, scaler, features
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")

# Load full dataset for charts
df_raw = pd.read_excel("Employees.xlsx")

st.title("💼 Salary Prediction Web App")
st.write("Predict salaries using machine learning + visualize trends.")

# ----------- USER INPUT FORM ------------
st.subheader("🔍 Enter Employee Details")

dept = st.selectbox("Department", df_raw["Department"].unique())
country = st.selectbox("Country", df_raw["Country"].unique())
center = st.selectbox("Center", df_raw["Center"].unique())

years = st.number_input("Years of Experience", 0, 40, 1)
job_rate = st.number_input("Job Rate", 0.5, 1.0, 1.0)
sick = st.number_input("Sick Leaves", 0, 50, 2)
unpaid = st.number_input("Unpaid Leaves", 0, 50, 0)
overtime = st.number_input("Overtime Hours", 0, 200, 10)

# Build input dict
input_data = {
    "Years": years,
    "Department": dept,
    "Country": country,
    "Center": center,
    "Job Rate": job_rate,
    "Sick Leaves": sick,
    "Unpaid Leaves": unpaid,
    "Overtime Hours": overtime,
}

# ----------- PREDICT BUTTON ------------
if st.button("Predict Salary"):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    # Ensure all features match training
    for col in feature_names:
        if col not in df:
            df[col] = 0

    df = df[feature_names]

    scaled = scaler.transform(df)
    predicted_salary = model.predict(scaled)[0]

    st.success(f"Predicted Annual Salary: ₹ {predicted_salary:,.2f}")

# ----------- ACTUAL VS PREDICTED CHART ------------
st.subheader("📊 Compare Actual vs Predicted Salaries")

df_temp = df_raw.copy()
df_temp["Predicted"] = model.predict(scaler.transform(
    pd.get_dummies(df_raw)[feature_names]
))

fig, ax = plt.subplots()
ax.scatter(df_temp["Annual Salary"], df_temp["Predicted"])
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title("Actual vs Predicted Salary")
st.pyplot(fig)

# ----------- TOP 20 EARNERS ------------
st.subheader("🏆 Top 20 Earners")

top20 = df_temp.nlargest(20, "Annual Salary")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(top20.index, top20["Annual Salary"])
ax2.set_title("Top 20 Actual Salaries")
st.pyplot(fig2)

