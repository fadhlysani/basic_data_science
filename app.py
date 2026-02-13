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
        st.error("Error: 'model_gaji.pkl' not found. Please ensure the model file is in the same directory.")
        st.stop()

    # Load the scaler
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: 'scaler.pkl' not found. Please ensure the scaler file is in the same directory.")
        st.stop()

    # Replicate LabelEncoders with all possible categories observed during training
    # IMPORTANT: The order of categories here must match the order used during training
    # If the original LabelEncoders were not saved, we must recreate them
    # precisely or explicitly map. For this example, we assume fitting on these ordered lists.

    # Example categories - replace with actual unique categories from your training data
    # and ensure their assigned integer values are consistent with training
    le_pendidikan = LabelEncoder()
    le_pendidikan.fit(['D3', 'S1', 'S2', 'SMA', 'SMK']) # Alphabetical order for consistent mapping
    
    le_jurusan = LabelEncoder()
    le_jurusan.fit(['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik']) # Alphabetical order

    return model, scaler, le_pendidikan, le_jurusan

model, scaler, le_pendidikan, le_jurusan = load_artifacts()

# --- 2. Define Feature Columns ---
feature_cols = [
    'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
]

# --- 3. User Input ---
st.header('Input Data Peserta')

col1, col2 = st.columns(2)
with col1:
    usia = st.slider('Usia', 18, 60, 30)
    durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 60)
    nilai_ujian = st.slider('Nilai Ujian', 50.0, 100.0, 75.0)

with col2:
    pendidikan = st.selectbox('Pendidikan Terakhir', ['SMA', 'SMK', 'D3', 'S1', 'S2'])
    jurusan = st.selectbox('Jurusan Pelatihan', ['Otomotif', 'Desain Grafis', 'Teknik Las', 'Teknik Listrik', 'Administrasi'])
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Wanita'])
    status_bekerja = st.selectbox('Status Bekerja', ['Sudah Bekerja', 'Belum Bekerja'])

# --- 4. Preprocessing Input ---
def preprocess_input(usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja):
    data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }
    df_input = pd.DataFrame([data])

    # Label Encoding
    df_input['Pendidikan'] = le_pendidikan.transform(df_input['Pendidikan'])
    df_input['Jurusan'] = le_jurusan.transform(df_input['Jurusan'])

    # One-Hot Encoding
    df_onehot = pd.get_dummies(df_input[['Jenis_Kelamin', 'Status_Bekerja']], prefix=['Jenis_Kelamin', 'Status_Bekerja'])
    df_onehot = df_onehot.astype(int)

    # Ensure all one-hot columns are present, fill with 0 if not
    expected_onehot_cols = [
        'Jenis_Kelamin_Laki-laki',
        'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja',
        'Status_Bekerja_Sudah Bekerja'
    ]
    for col in expected_onehot_cols:
        if col not in df_onehot.columns:
            df_onehot[col] = 0

    # Drop original categorical columns from df_input
    df_input.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'], inplace=True)

    # Concatenate numerical, label-encoded, and one-hot encoded features
    df_final_features = pd.concat([df_input, df_onehot], axis=1)

    # Reorder columns to match training order
    df_final_features = df_final_features[feature_cols]

    # Scaling
    input_scaled = scaler.transform(df_final_features)
    return input_scaled

# --- 5. Prediction ---
if st.button('Prediksi Gaji Awal'):
    processed_input = preprocess_input(usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja)
    prediction = model.predict(processed_input)[0]

    st.subheader('Hasil Prediksi:')
    st.success(f'Estimasi Gaji Awal: Rp {prediction:.2f} Juta')

st.markdown("""
---
**Catatan:**
Aplikasi ini memprediksi gaji awal berdasarkan model Machine Learning yang telah dilatih.
Hasil prediksi adalah estimasi dan dapat bervariasi.
""")
