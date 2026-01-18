import streamlit as st
import pandas as pd
import joblib

# Page config (helps avoid scrolling)
st.set_page_config(page_title="Iris Flower Prediction App", layout="wide")

# Title
st.title("🌸 Iris Flower Prediction Using Machine Learning")
st.subheader("Use the sliders below to enter flower measurements and predict the species.")

# Load trained model
model = joblib.load("model/iris_model.pkl")

# Create two columns
col1, col2 = st.columns(2)

# LEFT COLUMN → IMAGE
with col1:
    st.image(
        "iris_image.png",
        caption="Reference: Typical value ranges for Iris Setosa, Versicolor, Virginica",
        use_container_width=True
    )

# RIGHT COLUMN → SLIDERS
with col2:
    st.header("🔢 Enter Flower Measurements")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "SepalLengthCm": [sepal_length],
            "SepalWidthCm": [sepal_width],
            "PetalLengthCm": [petal_length],
            "PetalWidthCm": [petal_width]
        })

        prediction = model.predict(input_data)

        st.success(f"🌼 Predicted Iris Species: **{prediction[0]}**")
