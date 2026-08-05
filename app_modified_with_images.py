import os
import streamlit as st
import pickle
import numpy as np


# Banner image
st.image("images/banner.jpg", width="stretch")
st.markdown("""
<style>
.hero{
    background:linear-gradient(135deg,#2E8B57,#66BB6A);
    padding:30px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-top:10px;
    margin-bottom:25px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.2);
}
.info-card{
    background:#F5FFF5;
    padding:18px;
    border-radius:12px;
    border-left:5px solid #2E8B57;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🌱 Smart Crop Recommendation System</h1>
    <h4>AI-Powered Crop Prediction using Machine Learning</h4>
    <p>
    Predict the most suitable crop based on soil nutrients and weather conditions.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)


st.markdown("""
<style>
.main {background-color:#f4fff4;}
.stButton>button{
background:linear-gradient(90deg,#2E8B57,#4CAF50);
color:white;border:none;border-radius:10px;
height:50px;width:100%;font-size:18px;font-weight:bold;}
.result-box{
background:#1E1E1E;padding:15px;border-radius:10px;border-left:6px solid green;}
</style>
""", unsafe_allow_html=True)

st.title("🌱 Smart Crop Recommendation System")
st.caption("AI-powered crop prediction based on soil nutrients and weather conditions.")

@st.cache_resource
def load_artifacts():
    with open("New_RFmodel.pkl","rb") as f:
        model=pickle.load(f)
    with open("New_Scalar.pkl","rb") as f:
        scaler=pickle.load(f)
    return model,scaler

model, scaler = load_artifacts()

col1,col2=st.columns(2)

with col1:
    N=st.number_input("🌿 Nitrogen (N)",0.0,value=90.0)
    P=st.number_input("🧪 Phosphorus (P)",0.0,value=42.0)
    K=st.number_input("🍃 Potassium (K)",0.0,value=43.0)
    ph=st.slider("⚗ Soil pH",0.0,14.0,6.5)

with col2:
    temperature=st.slider("🌡 Temperature (°C)",0.0,50.0,20.0)
    humidity=st.slider("💧 Humidity (%)",0.0,100.0,80.0)
    rainfall=st.number_input("🌧 Rainfall (mm)",0.0,value=200.0)

input_values=[N,P,K,temperature,humidity,ph,rainfall]

crop_images = {
    "rice":"images/rice.jpg",
    "maize":"images/maize.jpg",
    "chickpea":"images/chickpea.webp",
    "kidneybeans":"images/kidneybeans.webp",
    "pigeonpeas":"images/pigeonpeas.jpg",
    "mothbeans":"images/mothbeans.webp",
    "mungbean":"images/mungbean.webp",
    "blackgram":"images/blackgram.webp",
    "lentil":"images/lentil.jpg",
    "pomegranate":"images/pomegranate.webp",
    "banana":"images/banana.jpg",
    "mango":"images/mango.jpg",
    "grapes":"images/grapes.jpg",
    "watermelon":"images/watermelon.jpg",
    "muskmelon":"images/muskmelon.jpg",
    "apple":"images/apple.jpg",
    "orange":"images/orange.jpg",
    "papaya":"images/papaya.jpg",
    "coconut":"images/coconut.jpg",
    "cotton":"images/cotton.jpg",
    "jute":"images/jute.jpg",
    "coffee":"images/coffee.jpg"
}


if st.button("🌾 Predict Crop"):
    input_array=np.array([input_values])
    scaled_input=scaler.transform(input_array)
    prediction=model.predict(scaled_input)

    confidence=None
    if hasattr(model,"predict_proba"):
        confidence=np.max(model.predict_proba(scaled_input))*100

    st.balloons()

    conf_html=f"<p><b>Confidence:</b> {confidence:.2f}%</p>" if confidence is not None else ""

    st.markdown(f"""
    <div class="result-box">
    <h3>🌾 Recommended Crop</h3>
    <h2 style="color:green;">{prediction[0]}</h2>
    {conf_html}
    </div>
    """, unsafe_allow_html=True)

    crop=str(prediction[0]).lower()
    if crop in crop_images:
        st.image(crop_images[crop], caption=prediction[0].title(), use_container_width=True)
