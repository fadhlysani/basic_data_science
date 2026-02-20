import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Prediksi Gaji Awal Lulusan Pelatihan Vokasi", layout="wide")

st.title("Aplikasi Prediksi Gaji Awal Lulusan Pelatihan Vokasi")

# --- 1. Muat Model dan Scaler ---
# Pastikan path file benar jika dijalankan di lingkungan yang berbeda
model_filename = 'gradient_boosting_model.pkl'
scaler_filename = 'standard_scaler.pkl'

try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    with open(scaler_filename, 'rb') as file:
        loaded_scaler = pickle.load(file)
    st.success("Model dan Scaler berhasil dimuat!")
except FileNotFoundError:
    st.error(f"Error: File model atau scaler tidak ditemukan. Pastikan '{model_filename}' dan '{scaler_filename}' berada di direktori yang sama dengan aplikasi ini.")
    st.stop() # Hentikan eksekusi aplikasi jika file tidak ditemukan

# --- 2. Re-inisialisasi Encoder (sesuai dengan proses training) ---
# Untuk deployment sejati, LabelEncoder instances dan one_hot_columns sebaiknya disimpan juga.
# Karena tidak bisa mengakses df_bersih langsung di Streamlit, kita perlu
# mendefinisikan ulang data unik atau menyimpan encoder itu sendiri.
# Untuk tujuan demonstrasi, kita akan hardcode nilai unik dari df_bersih
# (Idealnya, ini harus diambil dari data training asli atau encoder yang disimpan)

# Data unik dari df_bersih (diasumsikan sudah dikenal dari tahap EDA)
# Ganti dengan nilai unik aktual dari df_bersih.head() atau df_bersih.value_counts()

pendidikan_options = ['SMA', 'SMK', 'D3', 'S1', 'S2', 'SMP', 'SD'] # Contoh
jurusan_options = ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'] # Contoh
jenis_kelamin_options = ['Laki-laki', 'Wanita']
status_bekerja_options = ['Sudah Bekerja', 'Belum Bekerja']

# Re-fit LabelEncoder dengan semua kategori yang mungkin (dari df_bersih)
list_label_enc = ['Pendidikan', 'Jurusan']
label_encoders = {}

# Menggunakan set data yang lengkap dari df_bersih (jika df_bersih tersedia di sini, atau hardcode)
# Jika df_bersih tidak tersedia, kita harus memastikan semua kemungkinan kategori dicakup.
# Untuk demonstrasi ini, kita menggunakan options yang didefinisikan di atas.
le_pendidikan = LabelEncoder()
le_pendidikan.fit(pendidikan_options)
label_encoders['Pendidikan'] = le_pendidikan

le_jurusan = LabelEncoder()
le_jurusan.fit(jurusan_options)
label_encoders['Jurusan'] = le_jurusan


# Kolom one-hot encoding dan urutan fitur
list_one_hot = ['Jenis_Kelamin', 'Status_Bekerja']
# Ini harus sesuai dengan urutan kolom setelah One-Hot Encoding saat training
one_hot_columns = ['Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

# Urutan fitur akhir saat training (penting untuk konsistensi input model)
# Ini diambil dari variabel 'feature_cols' yang sudah ada di kernel
feature_cols_order = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                      'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                      'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

# --- 3. Fungsi untuk Preprocessing dan Prediksi Data Baru ---
def predict_new_data(new_raw_data, label_encoders, one_hot_columns, loaded_scaler, loaded_model, feature_cols_order):
    df_new = pd.DataFrame([new_raw_data])

    # Label Encoding
    df_label_processed = pd.DataFrame()
    for col in list_label_enc:
        if col in df_new.columns:
            df_label_processed[col] = label_encoders[col].transform(df_new[col].astype(str))
        else:
            df_label_processed[col] = 0 # Fallback

    # One-Hot Encoding
    df_onehot_processed = pd.get_dummies(df_new[list_one_hot], prefix=list_one_hot)
    df_onehot_processed = df_onehot_processed.astype(int)
    df_onehot_processed = df_onehot_processed.reindex(columns=one_hot_columns, fill_value=0)

    # Gabungkan fitur
    original_numerical_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian']
    df_processed = pd.concat([df_new[original_numerical_cols], df_label_processed, df_onehot_processed], axis=1)

    # Pastikan urutan kolom sesuai dengan training
    df_processed = df_processed[feature_cols_order]

    # Scaling
    df_scaled = pd.DataFrame(loaded_scaler.transform(df_processed), columns=df_processed.columns)

    # Prediksi
    prediction = loaded_model.predict(df_scaled)
    return prediction[0]

# --- 4. Streamlit UI untuk Input Pengguna ---
st.header("Input Data Calon Lulusan")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        usia = st.number_input("Usia (tahun)", min_value=18, max_value=60, value=25)
        durasi_jam = st.number_input("Durasi Pelatihan (jam)", min_value=20, max_value=100, value=40)
        nilai_ujian = st.number_input("Nilai Ujian (0-100)", min_value=0.0, max_value=100.0, value=85.0)
    
    with col2:
        pendidikan = st.selectbox("Pendidikan", options=pendidikan_options, index=pendidikan_options.index('S1'))
        jurusan = st.selectbox("Jurusan", options=jurusan_options, index=jurusan_options.index('Teknik Listrik'))

    with col3:
        jenis_kelamin = st.selectbox("Jenis Kelamin", options=jenis_kelamin_options, index=jenis_kelamin_options.index('Laki-laki'))
        status_bekerja = st.selectbox("Status Bekerja", options=status_bekerja_options, index=status_bekerja_options.index('Sudah Bekerja'))

    submit_button = st.form_submit_button("Prediksi Gaji Awal")

    if submit_button:
        new_data = {
            'Usia': usia,
            'Durasi_Jam': durasi_jam,
            'Nilai_Ujian': nilai_ujian,
            'Pendidikan': pendidikan,
            'Jurusan': jurusan,
            'Jenis_Kelamin': jenis_kelamin,
            'Status_Bekerja': status_bekerja,
        }

        predicted_salary = predict_new_data(new_data, label_encoders, one_hot_columns, loaded_scaler, loaded_model, feature_cols_order)

        st.success(f"Prediksi Gaji Awal untuk lulusan ini adalah: **{predicted_salary:.2f} Juta Rupiah**")
        st.balloons()
