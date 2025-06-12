import streamlit as st
import pandas as pd
import numpy as np
from model import load_data, train_model, predict

# Title of the app
st.title("Prediksi Cuaca Menggunakan Neural Network")

# Gantilah file CSV yang sudah ada di lokasi tertentu
data_file_path = "C:/Users/Dwi Ilham Ramadhany/Documents/Project ML/Prediksi_cuaca_dengan_JST-master/day.csv"  # Sesuaikan dengan path ke file CSV yang Anda miliki
dataset = load_data(data_file_path)

st.write("Data Cuaca yang Diupload", dataset.head())

# Train the model
weights = train_model(dataset)

# Inputs for user to predict the weather
st.sidebar.header("Masukkan Data Cuaca untuk Prediksi")

temp = st.number_input("Masukkan Nilai Temperature (°C)", min_value=-50, max_value=50)
hum = st.number_input("Masukkan Nilai Kelembapan (%)", min_value=0, max_value=100)
windspeed = st.number_input("Masukkan Nilai Kecepatan Angin (km/h)", min_value=0, max_value=150)

# When user clicks 'Predict'
if st.button("Prediksi Cuaca"):
    weather = predict(weights, temp, hum, windspeed)
    st.write(f"Prediksi Cuaca untuk Temperature: {temp}°C, Kelembapan: {hum}%, Kecepatan Angin: {windspeed} km/h adalah: {weather}")

# Footer
st.sidebar.markdown("Aplikasi Prediksi Cuaca menggunakan Neural Network")
