
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

st.set_page_config(layout="centered")
st.title('Aplikasi Prediksi Gaji Awal Lulusan Pelatihan Vokasi')

# --- 1. Load Model, Scaler, and Encoders ---
@st.cache_resource
def load_artifacts():
    # Load the trained model
    try:
        with open('model_gaji.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Model file 'model_gaji.pkl' not found. Please ensure the model is saved from your notebook.")
        st.stop()

    # Initialize and fit LabelEncoders with all unique categories from training data
    le_pendidikan = LabelEncoder()
    le_pendidikan.fit(['SMA', 'SMK', 'D3', 'S1', 'S2'])

    le_jurusan = LabelEncoder()
    le_jurusan.fit(['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Otomotif'])

    # Define the exact feature columns and their order used during training
    feature_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

    # Load the fitted StandardScaler
    try:
        with open('scaler.pkl', 'rb') as s_file:
            scaler = pickle.load(s_file)
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found. Please ensure the scaler is saved from your notebook.")
        st.stop()

    return model, le_pendidikan, le_jurusan, scaler, feature_cols

# Load all necessary artifacts
model, le_pendidikan, le_jurusan, scaler, feature_cols = load_artifacts()

# --- 2. Input Fields for Streamlit UI ---
st.header('Masukkan Data Peserta')

usia = st.slider('Usia', 18, 60, 30)
durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0)

pendidikan_options = ['SMA', 'SMK', 'D3', 'S1', 'S2']
pendidikan = st.selectbox('Pendidikan Terakhir', pendidikan_options)

jurusan_options = ['Otomotif', 'Desain Grafis', 'Teknik Las', 'Teknik Listrik', 'Administrasi']
jurusan = st.selectbox('Jurusan Pelatihan', jurusan_options)

jenis_kelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Wanita'])

status_bekerja = st.radio('Status Bekerja', ['Sudah Bekerja', 'Belum Bekerja'])


# --- 3. Preprocessing Function (Replicates notebook's preprocessing) ---
def preprocess_input(usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja,
                     le_pendidikan, le_jurusan, scaler, feature_cols):
    
    # Create a DataFrame from input data
    input_data = pd.DataFrame({
        'Usia': [usia],
        'Durasi_Jam': [durasi_jam],
        'Nilai_Ujian': [nilai_ujian],
        'Pendidikan': [pendidikan],
        'Jurusan': [jurusan],
        'Jenis_Kelamin': [jenis_kelamin],
        'Status_Bekerja': [status_bekerja]
    })

    df_processed = input_data.copy()

    # Apply Label Encoding
    df_processed['Pendidikan'] = le_pendidikan.transform(df_processed['Pendidikan'])
    df_processed['Jurusan'] = le_jurusan.transform(df_processed['Jurusan'])

    # Apply One-Hot Encoding
    list_one_hot_cols = ['Jenis_Kelamin', 'Status_Bekerja']
    df_onehot = pd.get_dummies(df_processed[list_one_hot_cols], prefix=list_one_hot_cols)
    df_onehot = df_onehot.astype(int) # Convert boolean to integer 0/1

    # Ensure all expected one-hot columns are present and in the correct order
    # This list must match the one-hot columns generated during training
    expected_onehot_cols = [
        'Jenis_Kelamin_Laki-laki',
        'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja',
        'Status_Bekerja_Sudah Bekerja'
    ]

    for col in expected_onehot_cols:
        if col not in df_onehot.columns:
            df_onehot[col] = 0
    df_onehot = df_onehot[expected_onehot_cols] # Ensure correct column order

    # Drop original categorical columns and concatenate one-hot encoded ones
    df_processed.drop(columns=list_one_hot_cols, inplace=True)
    df_final_features = pd.concat([df_processed, df_onehot], axis=1)

    # Reorder columns to exactly match the training feature order before scaling
    df_final_features = df_final_features[feature_cols]
    
    # Apply Standard Scaling
    scaled_features = scaler.transform(df_final_features)
    df_scaled_features = pd.DataFrame(scaled_features, columns=feature_cols) # Convert back to DataFrame with column names

    return df_scaled_features

# --- 4. Prediction Button and Display Result ---
if st.button('Prediksi Gaji Awal'):
    processed_input = preprocess_input(usia, durasi_jam, nilai_ujian, pendidikan, jurusan,
                                       jenis_kelamin, status_bekerja,
                                       le_pendidikan, le_jurusan, scaler, feature_cols)
    
    prediction = model.predict(processed_input)
    
    st.success(f'Prediksi Gaji Awal: Rp {prediction[0]:.2f} Juta')
