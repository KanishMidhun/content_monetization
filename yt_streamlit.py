import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load(r"C:\Users\kanis\OneDrive\Desktop\pythonvsworksdir\lux_house\youtube_revenue_model.pkl")

st.title("🎬 YouTube Ad Revenue Predictor")

# Input fields
views = st.number_input("Views", min_value=0)
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
watch_time = st.number_input("Watch Time (minutes)", min_value=0.0)
video_length = st.number_input("Video Length (minutes)", min_value=0.0)
subscribers = st.number_input("Subscribers", min_value=0)
#engagement_rate = st.number_input("Engagement Rate", min_value=0.0)
#revenue_per_view = st.number_input("Revenue per View ($)", min_value=0.0)
category = st.selectbox("Category", ["Gaming","Education","Tech","Entertainment","Lifestyle","Music"])
device = st.selectbox("Device", ["Mobile","Desktop","TV","Tablet"])
country = st.selectbox("Country", ["US","IN","CA","UK","DE","AU"])

# Derived feature
engagement_rate = (likes + comments) / views if views > 0 else 0

if st.button("Predict Revenue"):
    data = pd.DataFrame([[views, likes, comments, watch_time, video_length, subscribers,
                          category, device, country, engagement_rate,]],
                        columns=["views","likes","comments","watch_time_minutes",
                                 "video_length_minutes","subscribers","category",
                                 "device","country","engagement_rate"])
    pred = model.predict(data)[0]
    st.success(f"💰 Estimated Ad Revenue: ${pred:.2f}")
