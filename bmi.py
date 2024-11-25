import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class BMICalculatorML:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Smart BMI Monitoring & Rekomendasi")
        self.window.geometry("800x800")
        
        # Load dan preprocessing dataset
        self.load_dataset()
        
        # Setup model ML
        self.setup_ml_model()
        
        # GUI Components
        self.setup_gui()
    
    def load_dataset(self):
        """
        Load dataset dari file CSV
        Dataset yang dibutuhkan:
        - food_nutrition.csv: dataset nutrisi makanan
        - exercise_calories.csv: dataset aktivitas fisik
        """
        try:
            # Load dataset makanan (contoh struktur)
            # Columns: food_name, calories, protein, carbs, fats, category
            self.food_df = pd.read_csv('food_nutrition.csv')
            
            # Load dataset aktivitas (contoh struktur)
            # Columns: activity, calories_burned, intensity
            self.exercise_df = pd.read_csv('exercise_calories.csv')
            
        except FileNotFoundError:
            # Jika file tidak ditemukan, buat dummy dataset
            self.create_dummy_dataset()
    
    def create_dummy_dataset(self):
        """Membuat dummy dataset untuk demo"""
        # Dummy food dataset
        self.food_df = pd.DataFrame({
            'food_name': ['Nasi Putih', 'Ayam Dada', 'Sayur Bayam', 'Telur', 'Ikan Salmon'],
            'calories': [130, 165, 23, 155, 208],
            'protein': [2.7, 31, 2.9, 12.6, 22],
            'carbs': [28, 0, 3.6, 1.1, 0],
            'fats': [0.3, 3.6, 0.4, 10.6, 13],
            'category': ['Karbohidrat', 'Protein', 'Sayuran', 'Protein', 'Protein']
        })
        
        # Dummy exercise dataset
        self.exercise_df = pd.DataFrame({
            'activity': ['Jalan', 'Lari', 'Berenang', 'Bersepeda', 'Yoga'],
            'calories_burned': [150, 300, 400, 250, 150],
            'intensity': ['Rendah', 'Tinggi', 'Tinggi', 'Sedang', 'Rendah']
        })
    
    def setup_ml_model(self):
        """Setup dan training model ML"""
        # Contoh fitur untuk prediksi kalori harian
        X = np.random.rand(1000, 4)  # Simulasi dataset dengan fitur: [tinggi, berat, umur, aktivitas]
        y = 2000 + 15 * X[:, 0] + 25 * X[:, 1] - 10 * X[:, 2] + 100 * X[:, 3] + np.random.randn(1000) * 100
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale fitur
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
    
    def setup_gui(self):
        """Setup komponen GUI"""
        # Header
        tk.Label(self.window, text="Smart BMI Monitoring & Rekomendasi", 
                font=("Arial", 16, "bold")).pack(pady=10)
        
        # Input Frame
        input_frame = tk.Frame(self.window)
        input_frame.pack(pady=20)
        
        # Personal Info
        tk.Label(input_frame, text="Data Pribadi", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Tinggi
        tk.Label(input_frame, text="Tinggi (cm):").grid(row=1, column=0, padx=5, pady=5)
        self.tinggi_entry = tk.Entry(input_frame)
        self.tinggi_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Berat
        tk.Label(input_frame, text="Berat (kg):").grid(row=2, column=0, padx=5, pady=5)
        self.berat_entry = tk.Entry(input_frame)
        self.berat_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Umur
        tk.Label(input_frame, text="Umur:").grid(row=3, column=0, padx=5, pady=5)
        self.umur_entry = tk.Entry(input_frame)
        self.umur_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Level Aktivitas
        tk.Label(input_frame, text="Level Aktivitas:").grid(row=4, column=0, padx=5, pady=5)
        self.aktivitas_var = tk.StringVar(value="Sedang")
        aktivitas_options = ["Rendah", "Sedang", "Tinggi"]
        self.aktivitas_menu = ttk.Combobox(input_frame, textvariable=self.aktivitas_var, values=aktivitas_options)
        self.aktivitas_menu.grid(row=4, column=1, padx=5, pady=5)
        
        # Tombol Analisis
        tk.Button(self.window, text="Analisis Kesehatan", command=self.analisis_lengkap).pack(pady=10)
        
        # Frame Hasil
        result_frame = tk.Frame(self.window)
        result_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Labels untuk hasil
        self.bmi_label = tk.Label(result_frame, text="", font=("Arial", 12))
        self.bmi_label.pack(pady=5)
        
        self.kalori_label = tk.Label(result_frame, text="", font=("Arial", 12))
        self.kalori_label.pack(pady=5)
        
        self.rekomendasi_label = tk.Label(result_frame, text="", font=("Arial", 12), 
                                        wraplength=600, justify="left")
        self.rekomendasi_label.pack(pady=5)
        
        # Tabel rekomendasi makanan
        self.tree = ttk.Treeview(result_frame, columns=("Makanan", "Kalori", "Protein", "Karbohidrat", "Lemak"), 
                                show="headings", height=5)
        self.tree.pack(pady=10, fill="x")
        
        # Setup kolom tabel
        for col in ("Makanan", "Kalori", "Protein", "Karbohidrat", "Lemak"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
    
    def analisis_lengkap(self):
        """Melakukan analisis lengkap dengan ML"""
        try:
            # Ambil input
            tinggi = float(self.tinggi_entry.get())
            berat = float(self.berat_entry.get())
            umur = float(self.umur_entry.get())
            aktivitas = self.aktivitas_var.get()
            
            # Hitung BMI
            tinggi_m = tinggi / 100
            bmi = berat / (tinggi_m * tinggi_m)
            
            # Konversi aktivitas ke nilai numerik
            aktivitas_nilai = {"Rendah": 1, "Sedang": 2, "Tinggi": 3}[aktivitas]
            
            # Prediksi kebutuhan kalori dengan ML
            features = np.array([[tinggi, berat, umur, aktivitas_nilai]])
            features_scaled = self.scaler.transform(features)
            kalori_prediksi = self.model.predict(features_scaled)[0]
            
            # Update GUI dengan hasil
            self.bmi_label.config(text=f"BMI Anda: {bmi:.2f}")
            self.kalori_label.config(text=f"Kebutuhan Kalori Harian: {kalori_prediksi:.0f} kkal")
            
            # Generate rekomendasi makanan
            self.update_rekomendasi(bmi, kalori_prediksi)
            
        except ValueError:
            messagebox.showerror("Error", "Mohon masukkan angka yang valid!")
    
    def update_rekomendasi(self, bmi, kalori_target):
        """Update rekomendasi makanan berdasarkan BMI dan target kalori"""
        # Bersihkan tabel
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Filter makanan berdasarkan BMI
        if bmi < 18.5:
            makanan_rekomendasi = self.food_df[
                (self.food_df['calories'] >= 150) & 
                (self.food_df['protein'] >= 15)
            ].head()
        elif bmi < 25:
            makanan_rekomendasi = self.food_df[
                (self.food_df['calories'].between(100, 300)) & 
                (self.food_df['protein'] >= 10)
            ].head()
        else:
            makanan_rekomendasi = self.food_df[
                (self.food_df['calories'] <= 200) & 
                (self.food_df['protein'] >= 15)
            ].head()
        
        # Update tabel
        for _, row in makanan_rekomendasi.iterrows():
            self.tree.insert("", "end", values=(
                row['food_name'],
                f"{row['calories']} kkal",
                f"{row['protein']}g",
                f"{row['carbs']}g",
                f"{row['fats']}g"
            ))
        
        # Update label rekomendasi
        self.generate_rekomendasi_text(bmi, kalori_target)
    
    def generate_rekomendasi_text(self, bmi, kalori_target):
        """Generate teks rekomendasi berdasarkan BMI dan target kalori"""
        if bmi < 18.5:
            rekomendasi = f"""
Rekomendasi untuk meningkatkan berat badan sehat:
1. Target kalori harian: {kalori_target:.0f} kkal
2. Fokus pada makanan tinggi protein dan kalori sehat
3. Makan 5-6 kali sehari dengan porsi sedang
4. Tambahkan camilan sehat seperti kacang-kacangan
5. Latihan resistance training 2-3 kali seminggu
            """
        elif bmi < 25:
            rekomendasi = f"""
Rekomendasi untuk mempertahankan berat badan ideal:
1. Target kalori harian: {kalori_target:.0f} kkal
2. Pertahankan pola makan seimbang
3. Kombinasikan protein, karbohidrat kompleks, dan lemak sehat
4. Olahraga rutin 3-4 kali seminggu
5. Tidur cukup 7-8 jam sehari
            """
        else:
            rekomendasi = f"""
Rekomendasi untuk menurunkan berat badan:
1. Target kalori harian: {kalori_target:.0f} kkal
2. Fokus pada makanan tinggi protein dan serat
3. Batasi karbohidrat sederhana dan lemak jenuh
4. Olahraga kardio 4-5 kali seminggu
5. Catat asupan makanan harian
            """
        
        self.rekomendasi_label.config(text=rekomendasi)
    
    def run(self):
        """Jalankan aplikasi"""
        self.window.mainloop()

if __name__ == "__main__":
    app = BMICalculatorML()
    app.run()