
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Memuat model Gradient Boosting yang telah disimpan
with open('gradient_boosting_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Memuat scaler yang telah disimpan
with open('feature_scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Rekonstruksi mapping yang digunakan selama preprocessing
pendidikan_mapping = {'D3': 0, 'S1': 1, 'SMA': 2, 'SMK': 3}
jurusan_mapping = {'administrasi': 0, 'desain grafis': 1, 'otomotif': 2, 'teknik las': 3, 'teknik listrik': 4}

# Urutan kolom yang digunakan saat training (penting untuk prediksi)
feature_columns_order = ['Pendidikan', 'Jurusan', 'Jenis_Kelamin_Laki-laki',
                         'Jenis_Kelamin_Wanita', 'Status_Bekerja_Belum Bekerja',
                         'Status_Bekerja_Sudah Bekerja', 'Usia', 'Durasi_Jam', 'Nilai_Ujian']

def preprocess_input(input_data):
    df_input = pd.DataFrame([input_data])

    # Label Encoding
    df_input['Pendidikan'] = df_input['Pendidikan'].map(pendidikan_mapping)
    df_input['Jurusan'] = df_input['Jurusan'].map(jurusan_mapping)

    # One-Hot Encoding
    # Inisialisasi DataFrame dengan semua kolom one-hot diset ke 0
    processed_input = pd.DataFrame(0, index=[0], columns=feature_columns_order)

    # Isi nilai dari kolom yang sudah di-encode atau numerik
    processed_input['Pendidikan'] = df_input['Pendidikan']
    processed_input['Jurusan'] = df_input['Jurusan']
    processed_input['Usia'] = df_input['Usia']
    processed_input['Durasi_Jam'] = df_input['Durasi_Jam']
    processed_input['Nilai_Ujian'] = df_input['Nilai_Ujian']

    # Set 1 untuk kolom one-hot yang sesuai
    if df_input['Jenis_Kelamin'].iloc[0] == 'Laki-laki':
        processed_input['Jenis_Kelamin_Laki-laki'] = 1
    elif df_input['Jenis_Kelamin'].iloc[0] == 'Wanita':
        processed_input['Jenis_Kelamin_Wanita'] = 1

    if df_input['Status_Bekerja'].iloc[0] == 'Belum Bekerja':
        processed_input['Status_Bekerja_Belum Bekerja'] = 1
    elif df_input['Status_Bekerja'].iloc[0] == 'Sudah Bekerja':
        processed_input['Status_Bekerja_Sudah Bekerja'] = 1

    # Standard Scaling
    scaled_features = loaded_scaler.transform(processed_input)
    processed_input_scaled = pd.DataFrame(scaled_features, columns=feature_columns_order, index=processed_input.index)

    return processed_input_scaled

# --- Streamlit App --- #
st.set_page_config(page_title="Prediksi Gaji Pertama Peserta Pelatihan Vokasi", layout="centered")
st.title("Prediksi Gaji Pertama Peserta Pelatihan Vokasi")
st.markdown("Aplikasi ini memprediksi gaji pertama (dalam Juta Rupiah) peserta pelatihan vokasi berdasarkan beberapa fitur.")

st.subheader("Input Data Peserta")

# Input fields
jenis_kelamin = st.selectbox(
    'Jenis Kelamin',
    ('Laki-laki', 'Wanita')
)

usia = st.slider(
    'Usia (Tahun)',
    min_value=18.0, max_value=60.0, value=25.0, step=1.0
)

pendidikan = st.selectbox(
    'Pendidikan Terakhir',
    ('SMA', 'SMK', 'D3', 'S1')
)

jurusan = st.selectbox(
    'Jurusan Pelatihan',
    ('administrasi', 'desain grafis', 'otomotif', 'teknik las', 'teknik listrik')
)

durasi_jam = st.slider(
    'Durasi Pelatihan (Jam)',
    min_value=20, max_value=100, value=50, step=1
)

nilai_ujian = st.slider(
    'Nilai Ujian Akhir',
    min_value=50.0, max_value=100.0, value=75.0, step=0.1
)

status_bekerja = st.selectbox(
    'Status Setelah Pelatihan',
    ('Belum Bekerja', 'Sudah Bekerja')
)

# Prepare input for prediction
input_data = {
    'Jenis_Kelamin': jenis_kelamin,
    'Usia': usia,
    'Pendidikan': pendidikan,
    'Jurusan': jurusan,
    'Durasi_Jam': durasi_jam,
    'Nilai_Ujian': nilai_ujian,
    'Status_Bekerja': status_bekerja
}

if st.button('Prediksi Gaji Pertama'):
    processed_input = preprocess_input(input_data)
    predicted_gaji = loaded_model.predict(processed_input)[0]
    
    st.success(f"Prediksi Gaji Pertama: **{predicted_gaji:.2f} Juta Rupiah**")

st.markdown("""
---
### Cara Menjalankan Aplikasi Streamlit ini:
1. Pastikan Anda telah menyimpan `gradient_boosting_model.pkl` dan `feature_scaler.pkl` di direktori yang sama dengan skrip Streamlit ini.
2. Simpan kode di atas sebagai `app.py` (atau nama file Python lainnya). (Sudah dilakukan secara otomatis oleh sel ini)
3. Buka terminal atau command prompt di lingkungan Anda.
4. Navigasi ke direktori tempat Anda menyimpan `app.py`.
5. Jalankan perintah: `streamlit run app.py`
6. Aplikasi akan terbuka di browser web Anda.
""")
