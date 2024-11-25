import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load dataset dan model yang telah disimpan
with open("food_group_model.pkl", "rb") as file:
    model = pickle.load(file)

# Fungsi untuk menghitung BMI dan kebutuhan nutrisi
def hitung_bmi_dan_kebutuhan(berat, tinggi, usia, jenis_kelamin, tingkat_aktivitas):
    tinggi_m = tinggi / 100
    bmi = berat / (tinggi_m ** 2)

    # Tentukan kategori BMI
    if bmi < 18.5:
        kategori_bmi = 'Underweight'
    elif 18.5 <= bmi < 25:
        kategori_bmi = 'Normal'
    elif 25 <= bmi < 30:
        kategori_bmi = 'Overweight'    
    else:
        kategori_bmi = 'Obesitas'

    # Hitung BMR (Tingkat Metabolisme Basal)
    if jenis_kelamin.lower() == 'l':  # 'l' untuk laki-laki
        bmr = 88.362 + (13.397 * berat) + (4.799 * tinggi) - (5.677 * usia)
    else:  # 'p' untuk perempuan
        bmr = 447.593 + (9.247 * berat) + (3.098 * tinggi) - (4.330 * usia)

    # Faktor aktivitas
    faktor_aktivitas = {
        'Sedentary': 1.2,
        'Light': 1.375,
        'Moderate': 1.55,
        'Active': 1.725,
        'Very active': 1.9
    }

    # Hitung kebutuhan kalori harian
    kalori_harian = bmr * faktor_aktivitas[tingkat_aktivitas]
    kebutuhan_protein = berat * 0.8  # 0.8g per kg berat badan
    kebutuhan_lemak = (kalori_harian * 0.25) / 9  # 25% dari kalori untuk lemak
    kebutuhan_karbohidrat = (kalori_harian * 0.55) / 4  # 55% dari kalori untuk karbohidrat

    return {
        'BMI': round(bmi, 2),
        'Kategori': kategori_bmi,
        'Kalori Harian (kcal)': round(kalori_harian),
        'Protein (g)': round(kebutuhan_protein),
        'Lemak (g)': round(kebutuhan_lemak),
        'Karbohidrat (g)': round(kebutuhan_karbohidrat)
    }

# Fungsi untuk rekomendasi makanan
def rekomendasi_makanan(kategori_bmi, kalori_harian, pantangan=None):
    df = pd.read_csv("combined_nutrition_dataset.csv")  # Load dataset gabungan
    filtered_df = df.copy()
    
    kolom_serat = 'Fiber_g'
    kolom_vitamin_c = 'VitC_mg' 
    kolom_kalsium = 'Calcium_mg'
    kolom_kalori = 'Energy_kcal'
    kolom_protein = 'Protein_g'

    # Filter berdasarkan pantangan diet
    if pantangan:
        for item in pantangan:
            filtered_df = filtered_df[~filtered_df['KelompokMakanan'].str.contains(item, na=False)]

    # Sesuaikan rekomendasi berdasarkan kategori BMI
    if kategori_bmi == 'Underweight':
        filtered_df['skor'] = (filtered_df[kolom_kalori] / 500 + filtered_df[kolom_protein] / 20)
    elif kategori_bmi in ['Overweight', 'O  besitas']:
        filtered_df['skor'] = (filtered_df[kolom_protein] / 20 + filtered_df[kolom_serat] / 10 - filtered_df[kolom_kalori] / 1000)
    else:
        filtered_df['skor'] = (filtered_df[kolom_protein] / 20 + filtered_df[kolom_serat] / 10 + filtered_df[kolom_vitamin_c] / 60 + filtered_df[kolom_kalsium] / 1000) / 4

    # Pilih 5 rekomendasi terbaik
    rekomendasi = filtered_df.nlargest(5, 'skor')
    return rekomendasi[['Descrip', 'FoodGroup', 'Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g', 'Fiber_g']]

    # return rekomendasi[['Deskripsi', 'KelompokMakanan', 'Kalori', 'Protein', 'Lemak', 'Karbohidrat', 'Serat']]

# Streamlit UI
st.title("Kalkulator BMI dan Rekomendasi Makanan")

# Input data pengguna
berat = st.number_input("Berat Badan (kg)", min_value=1, max_value=300, value=70)
tinggi = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170)
usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=30)
jenis_kelamin = st.selectbox("Jenis Kelamin", options=["Laki-laki (L)", "Perempuan (P)"], index=0)
tingkat_aktivitas = st.selectbox("Tingkat Aktivitas", options=[
    'Sedentary', 'Light', 'Moderate', 'Active', 'Very active'], index=0)
pantangan = st.multiselect("Pantangan Makanan (opsional)", options=[
    "Susu", "Gluten", "Kacang", "Makanan Laut", "Telur"])

# Tombol untuk prediksi
if st.button("Hitung dan Rekomendasikan"):
    hasil = hitung_bmi_dan_kebutuhan(berat, tinggi, usia, jenis_kelamin[0].lower(), tingkat_aktivitas)

    # Tampilkan hasil BMI dan kebutuhan nutrisi
    st.subheader("Hasil BMI dan Kebutuhan Nutrisi")
    col1, col2, col3 = st.columns(3)

    col1.metric("BMI", hasil['BMI'])
    col2.metric("Kategori", hasil['Kategori'])
    col3.metric("Kalori Harian", f"{hasil['Kalori Harian (kcal)']} kcal")

    st.write("**Rincian Kebutuhan Nutrisi:**")
    st.write(f"- **Protein:** {hasil['Protein (g)']} g")
    st.write(f"- **Lemak:** {hasil['Lemak (g)']} g")
    st.write(f"- **Karbohidrat:** {hasil['Karbohidrat (g)']} g")

    # Tampilkan rekomendasi makanan
    st.subheader("Rekomendasi Makanan")
    rekomendasi = rekomendasi_makanan(hasil['Kategori'], hasil['Kalori Harian (kcal)'], pantangan)
    st.table(rekomendasi)
