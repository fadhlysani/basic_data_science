import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Load Model and Scaler ---
# Asumsi 'model_gaji.pkl' dan 'scaler.pkl' ada di direktori yang sama
with open('model_gaji.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# --- 2. Define Feature Columns and Expected One-Hot Columns (must match training) ---
feature_cols = [
    'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
]

expected_onehot_cols = [
    'Jenis_Kelamin_Laki-laki',
    'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja',
    'Status_Bekerja_Sudah Bekerja'
]

# --- 3. Preprocessing Function (from previous steps) ---
def preprocess_input(raw_input_data):
    df_raw_input = pd.DataFrame(raw_input_data)
    df_processed_temp = df_raw_input.copy()

    le_pendidikan = LabelEncoder()
    # Fit dengan semua kategori pendidikan yang mungkin dari data pelatihan
    le_pendidikan.fit(['SMA', 'SMK', 'D3', 'S1', 'S2'])
    df_processed_temp['Pendidikan'] = le_pendidikan.transform(df_processed_temp['Pendidikan'])

    le_jurusan = LabelEncoder()
    # Fit dengan semua kategori jurusan yang mungkin dari data pelatihan
    le_jurusan.fit(['Otomotif', 'Desain Grafis', 'Teknik Las', 'Teknik Listrik', 'Administrasi'])
    df_processed_temp['Jurusan'] = le_jurusan.transform(df_processed_temp['Jurusan'])

    list_one_hot_cols_to_process = ['Jenis_Kelamin', 'Status_Bekerja']
    df_onehot_part = pd.get_dummies(df_processed_temp[list_one_hot_cols_to_process], prefix=list_one_hot_cols_to_process)
    df_onehot_part = df_onehot_part.astype(int)

    for col in expected_onehot_cols:
        if col not in df_onehot_part.columns:
            df_onehot_part[col] = 0
    df_onehot_part = df_onehot_part[expected_onehot_cols] # Pastikan urutan kolom sesuai

    df_processed_temp.drop(columns=list_one_hot_cols_to_process, inplace=True)
    df_final_features_merged = pd.concat([df_processed_temp, df_onehot_part], axis=1)

    df_final_features_ordered = df_final_features_merged[feature_cols] # Pastikan urutan kolom sesuai

    input_scaled_array = loaded_scaler.transform(df_final_features_ordered)
    input_scaled_df = pd.DataFrame(input_scaled_array, columns=feature_cols)
    return input_scaled_df

# --- 4. Streamlit Application Layout ---
st.title('💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi')
st.write('Aplikasi ini memprediksi gaji awal seorang lulusan pelatihan vokasi berdasarkan beberapa faktor.')

# User Inputs
st.sidebar.header('Input Data Peserta')

usia = st.sidebar.slider('Usia (Tahun)', min_value=18, max_value=60, value=25)
durasi_jam = st.sidebar.slider('Durasi Pelatihan (Jam)', min_value=20, max_value=100, value=60)
nilai_ujian = st.sidebar.slider('Nilai Ujian', min_value=50.0, max_value=100.0, value=75.0, step=0.1)
pendidikan = st.sidebar.selectbox('Pendidikan', ['SMA', 'SMK', 'D3', 'S1', 'S2'])
jurusan = st.sidebar.selectbox('Jurusan', ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'])
jenis_kelamin = st.sidebar.selectbox('Jenis Kelamin', ['Laki-laki', 'Wanita'])
status_bekerja = st.sidebar.selectbox('Status Bekerja', ['Sudah Bekerja', 'Belum Bekerja'])

# Collect raw input data into a dictionary
raw_input_data = {
    'Usia': [usia],
    'Durasi_Jam': [durasi_jam],
    'Nilai_Ujian': [nilai_ujian],
    'Pendidikan': [pendidikan],
    'Jurusan': [jurusan],
    'Jenis_Kelamin': [jenis_kelamin],
    'Status_Bekerja': [status_bekerja]
}

# --- 5. Make Prediction ---
if st.sidebar.button('Prediksi Gaji'):
    try:
        processed_input = preprocess_input(raw_input_data)
        prediction = loaded_model.predict(processed_input)[0]

        st.subheader('Hasil Prediksi Gaji Awal:')
        st.success(f'Gaji awal yang diprediksi adalah: **Rp {prediction:.2f} Juta**')
        st.write('Ini adalah estimasi gaji awal berdasarkan model kami.')

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.error("Pastikan semua input sudah sesuai dan model serta scaler dimuat dengan benar.")
