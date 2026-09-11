import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# 1. Load the trained model + the artifacts saved during training
#    (see save_artifacts_snippet.py — run that once after training)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("lr_final_model.pkl")
    model_columns = joblib.load("model_columns.pkl")   # exact 52 training columns, in order
    categories = joblib.load("categories.pkl")          # dict: {col_name: [unique values]}
    return model, model_columns, categories

model, model_columns, categories = load_artifacts()

st.title("🚗 Car Price Predictor")
st.write("Enter the car's details below to estimate its price.")

# ------------------------------------------------------------------
# 2. Build the input form.
#    Categorical fields use dropdowns populated from categories.pkl
#    so the user can never type a brand/model the model never saw.
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", categories["Brand"])
    model_name = st.selectbox("Model", categories["Model"])
    year = st.number_input("Year", min_value=1990, max_value=2026, value=2020, step=1)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    fuel_type = st.selectbox("Fuel Type", categories["Fuel_Type"])

with col2:
    transmission = st.selectbox("Transmission", categories["Transmission"])
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000, step=1000)
    doors = st.number_input("Doors", min_value=2, max_value=5, value=4, step=1)
    owner_count = st.number_input("Owner Count", min_value=1, max_value=10, value=1, step=1)

# ------------------------------------------------------------------
# 3. Predict button
# ------------------------------------------------------------------
if st.button("Predict Price"):

    # 3a. Raw single-row dataframe, same column names as the ORIGINAL
    #     (pre-encoding) training dataframe.
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Model": model_name,
        "Year": year,
        "Engine_Size": engine_size,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Mileage": mileage,
        "Doors": doors,
        "Owner_Count": owner_count,
    }])

    # 3b. One-hot encode this single row the same way training data was encoded.
    input_encoded = pd.get_dummies(input_df)

    # 3c. THE KEY STEP: force this row into the exact same columns the
    #     model was trained on. Any column the model expects but this
    #     row doesn't have (e.g. Brand_Kia when user picked Toyota) is
    #     filled with 0. Any stray column is dropped. Order is fixed too.
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    # 3d. Predict
    predicted_price = model.predict(input_final)[0]

    st.success(f"💰 Estimated Price: {predicted_price:,.2f}")

    with st.expander("See the exact feature row sent to the model"):
        st.dataframe(input_final)