import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dan persiapkan data
def load_and_prepare_data():
    # Load dataset
    df = pd.read_csv('test.csv')
    
    # Menampilkan informasi dataset
    print("Dataset Info:")
    print(df.info())
    
    # Menampilkan statistik dasar
    print("\nDataset Statistics:")
    print(df.describe())
    
    return df

# 2. Fungsi untuk mengelompokkan makanan berdasarkan kebutuhan nutrisi
def categorize_food_by_nutrition(df):
    """
    Mengelompokkan makanan berdasarkan kandungan nutrisi utama
    """
    df['NutritionCategory'] = 'Balanced'
    
    # High Protein Foods (>20% daily value)
    mask_protein = df['Protein_g'] > 20
    df.loc[mask_protein, 'NutritionCategory'] = 'High Protein'
    
    # High Carb Foods (>30% daily value)
    mask_carb = df['Carb_g'] > 30
    df.loc[mask_carb, 'NutritionCategory'] = 'High Carb'
    
    # High Fiber Foods (>25% daily value)
    mask_fiber = df['Fiber_g'] > 25
    df.loc[mask_fiber, 'NutritionCategory'] = 'High Fiber'
    
    return df

# 3. Fungsi untuk menghitung skor nutrisi makanan
def calculate_nutrition_score(row, bmi_category):
    """
    Menghitung skor kesesuaian nutrisi berdasarkan BMI
    """
    score = 0
    
    if bmi_category == 'Underweight':
        # Untuk underweight, prioritaskan kalori dan protein
        score += (row['Energy_kcal'] / 2000) * 3
        score += (row['Protein_g'] / 50) * 2
        score += (row['Fat_g'] / 65) * 1
        
    elif bmi_category == 'Overweight' or bmi_category == 'Obese':
        # Untuk overweight/obese, prioritaskan protein dan serat, kurangi kalori
        score += (row['Protein_g'] / 50) * 2
        score += (row['Fiber_g'] / 25) * 2
        score -= (row['Energy_kcal'] / 2000) * 1
        score -= (row['Sugar_g'] / 25) * 2
        
    else:  # Normal weight
        # Untuk berat normal, seimbangkan semua nutrisi
        score += (row['Protein_g'] / 50)
        score += (row['Fiber_g'] / 25)
        score += (row['VitC_mg'] / 60)
        score += (row['Calcium_mg'] / 1000)
        score = score / 4  # Normalize
    
    return max(0, score)  # Ensure non-negative score

# 4. Fungsi untuk membuat rekomendasi makanan
def get_food_recommendations(df, user_data):
    """
    Membuat rekomendasi makanan berdasarkan BMI dan kebutuhan nutrisi
    """
    bmi = calculate_bmi(user_data['weight'], user_data['height'])
    bmi_category = get_bmi_category(bmi)
    
    # Hitung kebutuhan kalori harian
    daily_calories = calculate_daily_calories(user_data)
    
    # Hitung skor nutrisi untuk setiap makanan
    df['NutritionScore'] = df.apply(lambda x: calculate_nutrition_score(x, bmi_category), axis=1)
    
    # Filter makanan berdasarkan preferensi dan batasan
    suitable_foods = filter_foods(df, user_data)
    
    # Buat rekomendasi untuk setiap waktu makan
    recommendations = {
        'breakfast': get_meal_recommendations(suitable_foods, daily_calories * 0.3),
        'lunch': get_meal_recommendations(suitable_foods, daily_calories * 0.4),
        'dinner': get_meal_recommendations(suitable_foods, daily_calories * 0.3)
    }
    
    return recommendations

# 5. Helper functions
def calculate_bmi(weight, height):
    """
    Menghitung BMI
    weight: berat dalam kg
    height: tinggi dalam cm
    """
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return bmi

def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def calculate_daily_calories(user_data):
    """
    Menghitung kebutuhan kalori harian berdasarkan Harris-Benedict Equation
    """
    # Basic BMR calculation
    if user_data['gender'] == 'M':
        bmr = 88.362 + (13.397 * user_data['weight']) + \
              (4.799 * user_data['height']) - (5.677 * user_data['age'])
    else:
        bmr = 447.593 + (9.247 * user_data['weight']) + \
              (3.098 * user_data['height']) - (4.330 * user_data['age'])
    
    # Activity factor
    activity_factors = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    
    return bmr * activity_factors[user_data['activity_level']]

def filter_foods(df, user_data):
    """
    Filter makanan berdasarkan preferensi dan batasan pengguna
    """
    filtered_df = df.copy()
    
    # Filter berdasarkan batasan kalori
    max_calories_per_meal = calculate_daily_calories(user_data) * 0.4  # Max 40% kalori per makanan
    filtered_df = filtered_df[filtered_df['Energy_kcal'] <= max_calories_per_meal]
    
    # Filter berdasarkan preferensi diet
    if 'vegetarian' in user_data['diet_preferences']:
        filtered_df = filtered_df[~filtered_df['FoodGroup'].str.contains('Meat|Fish|Poultry', na=False)]
    
    return filtered_df

def get_meal_recommendations(df, target_calories):
    """
    Mendapatkan rekomendasi makanan untuk satu waktu makan
    """
    # Sort by nutrition score
    df_sorted = df.sort_values('NutritionScore', ascending=False)
    
    # Get top recommendations that fit calorie target
    recommendations = []
    current_calories = 0
    
    for _, food in df_sorted.iterrows():
        if current_calories + food['Energy_kcal'] <= target_calories:
            recommendations.append({
                'name': food['Descrip'],
                'calories': food['Energy_kcal'],
                'protein': food['Protein_g'],
                'carbs': food['Carb_g'],
                'fat': food['Fat_g'],
                'fiber': food['Fiber_g']
            })
            current_calories += food['Energy_kcal']
            
            if len(recommendations) >= 3:  # Maksimum 3 item per waktu makan
                break
    
    return recommendations

# 6. Main function untuk menjalankan sistem
def main():
    # Load data
    df = load_and_prepare_data()
    
    # Kategorikan makanan
    df = categorize_food_by_nutrition(df)
    
    # Contoh data pengguna
    user_data = {
        'age': 30,
        'weight': 70,
        'height': 170,
        'gender': 'M',
        'activity_level': 'moderate',
        'diet_preferences': ['balanced']
    }
    
    # Dapatkan rekomendasi
    recommendations = get_food_recommendations(df, user_data)
    
    return df, recommendations

if __name__ == "__main__":
    df, recommendations = main()