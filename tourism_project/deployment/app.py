import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------- Load Model & Encoders ----------------
model_path = hf_hub_download(
    repo_id="DIVHF/tourism-package-model",
    filename="best_tourism_package_model_v1.joblib"
)
model = joblib.load(model_path)

encoders_path = hf_hub_download(
    repo_id="DIVHF/Tourism-AML-MLOps",
    filename="label_encoders.pkl",
    repo_type="dataset"
)
label_encoders = joblib.load(encoders_path)

# ---------------- Streamlit UI ----------------
st.title("Tourism Package Purchase Prediction")
st.write("Fill the customer details below to predict if they'll purchase a travel package.")

# Input fields
Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips", 1, 20, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 5, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=300000.0)

# ---------------- Prepare Input DataFrame ----------------
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# ---------------- Preprocessing ----------------
# Merge Free Lancer into Small Business
if input_data.loc[0, 'Occupation'] == "Free Lancer":
    input_data.loc[0, 'Occupation'] = "Small Business"

# Apply label encoders
for col, le in label_encoders.items():
    if col in input_data.columns:
        # Safely transform unseen categories to first class
        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

# ---------------- Prediction ----------------
classification_threshold = 0.45

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    result = "WILL purchase the travel package" if pred == 1 else "is UNLIKELY to purchase"

    st.subheader("🔮 Prediction Result")
    st.write(f"Customer **{result}**")
    st.write(f"Probability of buying: **{prob:.2f}**")
