import streamlit as st
import pandas as pd
from weather_model import load_data, train_model, predict_weather

# Streamlit app title
st.title("Cricket Weather Predictor")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Sidebar inputs
temperature = st.sidebar.slider('Temperature (°C)', min_value=0, max_value=50, value=30)
humidity = st.sidebar.slider('Humidity (%)', min_value=0, max_value=100, value=50)
windspeed = st.sidebar.slider('Wind Speed (km/h)', min_value=0, max_value=50, value=10)

# File upload for weather data
uploaded_file = st.file_uploader("Upload your CSV file with weather data", type=["csv"])

if uploaded_file is not None:
    # Load and display data
    data = load_data(uploaded_file)
    st.write("Weather Data:", data)

    # Train the model using the uploaded data
    model = train_model(data)
    
    # Predict based on user inputs
    st.write("Input Values:")
    st.write(f"Temperature: {temperature} °C, Humidity: {humidity}%, Wind Speed: {windspeed} km/h")

    prediction = predict_weather(model, temperature, humidity, windspeed)
    
    # Display the prediction result
    st.write(f"Predicted Rain Probability: {'Rain' if prediction > 0.5 else 'No Rain'}")

else:
    st.write("Please upload a CSV file to get started.")
