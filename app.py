import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def predict_bmi_category(weight, height, gender):
    try:
        # Calculate BMI first
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        # Load model
        with open("bmi_classification_model.pkl", "rb") as file:
            bmi_model = pickle.load(file)
            
        # Prepare input features
        input_features = pd.DataFrame([[1 if gender.lower() == 'l' else 0, height, weight]], 
                                    columns=['Gender', 'Height', 'Weight'])
        
        # Make prediction using model's predict method
        prediction = int(bmi_model.predict(input_features)[0])
        
        # Map prediction to category
        bmi_categories = {
            1: 'Underweight',
            2: 'Normal',
            3: 'Overweight',
            4: 'Obese'
        }
        
        category = bmi_categories.get(prediction, 'Unknown')
        return category, round(bmi, 2)
    
    except Exception as e:
        print(f"Detailed error: {e}")  # For debugging
        # Fallback to basic BMI calculation if model fails
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        if bmi < 18.5:
            category = 'Underweight'
        elif 18.5 <= bmi < 25:
            category = 'Normal'
        elif 25 <= bmi < 30:
            category = 'Overweight'    
        else:
            category = 'Obese'
            
        return category, round(bmi, 2)

def calculate_bmi_and_needs(weight, height, age, gender, activity_level):
    height_m = height / 100
    
    # Get BMI category from model
    bmi_category, bmi = predict_bmi_category(weight, height, gender)
    
    # Calculate BMR
    if gender.lower() == 'l':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    
    # Activity factor
    activity_factors = {
        'Sedentary': 1.2,
        'Light': 1.375,
        'Moderate': 1.55,
        'Active': 1.725,
        'Very active': 1.9
    }
    
    # Calculate daily needs
    daily_calories = bmr * activity_factors[activity_level]
    protein_needs = weight * 0.8
    fat_needs = (daily_calories * 0.25) / 9
    carb_needs = (daily_calories * 0.55) / 4
    
    return {
        'BMI': bmi,
        'Category': bmi_category,
        'Daily Calories (kcal)': round(daily_calories),
        'Protein (g)': round(protein_needs),
        'Fat (g)': round(fat_needs),
        'Carbohydrate (g)': round(carb_needs)
    }

# [Rest of the code remains unchanged - recommend_food function and main function stay the same]
def recommend_food(nutritional_needs, model, scaler, restrictions=None):
    try:
        # Load dataset
        df = pd.read_csv("combined_nutrition_dataset.csv")
        
        # Handle missing values for numerical features
        features = ['Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g', 'Fiber_g', 'VitC_mg', 'Calcium_mg']
        df[features] = df[features].fillna(df[features].mean())

        # Handle missing values for text features
        df['Descrip'] = df['Descrip'].fillna("Tidak Diketahui")
        df['FoodGroup'] = df['FoodGroup'].fillna("Tidak Diketahui")

        # Filter based on BMI category
        bmi_category = nutritional_needs['Category']
        
        # Adjust scoring based on BMI category
        if bmi_category == 'Underweight':
            df['category_score'] = (
                (df['Energy_kcal'] / nutritional_needs['Daily Calories (kcal)']) * 0.4 +
                (df['Protein_g'] / nutritional_needs['Protein (g)']) * 0.4 +
                (df['Carb_g'] / nutritional_needs['Carbohydrate (g)']) * 0.2
            )
        elif bmi_category in ['Overweight', 'Obese']:
            df['category_score'] = (
                (1 - abs(df['Energy_kcal'] - nutritional_needs['Daily Calories (kcal)']) / nutritional_needs['Daily Calories (kcal)']) * 0.3 +
                (df['Protein_g'] / nutritional_needs['Protein (g)']) * 0.4 +
                (df['Fiber_g'] / df['Carb_g']) * 0.3
            )
        else:  # Normal
            df['category_score'] = (
                (1 - abs(df['Energy_kcal'] - nutritional_needs['Daily Calories (kcal)']) / nutritional_needs['Daily Calories (kcal)']) * 0.4 +
                (1 - abs(df['Protein_g'] - nutritional_needs['Protein (g)']) / nutritional_needs['Protein (g)']) * 0.3 +
                (1 - abs(df['Carb_g'] - nutritional_needs['Carbohydrate (g)']) / nutritional_needs['Carbohydrate (g)']) * 0.3
            )
        
        # Handle dietary restrictions
        if restrictions:
            for restriction in restrictions:
                df = df[~df['FoodGroup'].str.contains(restriction, na=False)]
        
        # Prepare features for model prediction
        X = df[features]
        X_scaled = scaler.transform(X)
        
        # Get model predictions and probabilities
        predictions = model.predict_proba(X_scaled)
        df['model_score'] = predictions[:, -1]  # Probability of highest category
        
        # Calculate final score combining model prediction and category-based score
        df['final_score'] = (df['model_score'] * 0.5) + (df['category_score'] * 0.5)
        
        # Filter out rows where 'Descrip' or 'FoodGroup' is "Tidak Diketahui"
        df = df[(df['Descrip'] != "Tidak Diketahui") & (df['FoodGroup'] != "Tidak Diketahui")]
        
        # Get top 20 recommendations based on final score
        top_20 = df.nlargest(20, 'final_score')
        
        # Randomly select 5 from top 20
        recommendations = top_20.sample(n=5, random_state=None)
        
        # Calculate accuracy score for each recommendation
        accuracy_scores = recommendations['final_score'] * 100
        
        # Add accuracy score to the output
        result = recommendations[['Descrip', 'FoodGroup', 'Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g', 'Fiber_g']].copy()
        result['Accuracy'] = accuracy_scores.round(2).astype(str) + '%'
        
        return result
        
    except Exception as e:
        st.error(f"Error in food recommendation: {str(e)}")
        print(f"Error details: {e}")
        return None

def main():
    st.title("BMI Calculator and Food Recommendation System")
    
    # Try to load the food recommendation model
    try:
        with open("food_recommendation_model.pkl", "rb") as file:
            model, scaler = pickle.load(file)
    except FileNotFoundError:
        st.error("Food recommendation model not found!")
        return
    
    # User inputs
    weight = st.number_input("Weight (kg)", min_value=1, max_value=300, value=70)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", options=["Male (L)", "Female (P)"], index=0)
    activity_level = st.selectbox("Activity Level", 
                                options=['Sedentary', 'Light', 'Moderate', 'Active', 'Very active'],
                                index=0)
    restrictions = st.multiselect("Dietary Restrictions", 
                                options=["Dairy", "Gluten", "Nuts", "Seafood", "Eggs"])
    
    if st.button("Calculate and Recommend"):
        # Calculate nutritional needs
        needs = calculate_bmi_and_needs(weight, height, age, gender[0].lower(), activity_level)
        
        # Display BMI and nutritional needs
        st.subheader("BMI and Nutritional Needs")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("BMI", needs['BMI'])
        col2.metric("Category", needs['Category'])
        col3.metric("Daily Calories", f"{needs['Daily Calories (kcal)']} kcal")
        
        st.write("**Detailed Nutritional Needs:**")
        st.write(f"- **Protein:** {needs['Protein (g)']} g")
        st.write(f"- **Fat:** {needs['Fat (g)']} g")
        st.write(f"- **Carbohydrate:** {needs['Carbohydrate (g)']} g")
        
        # Get and display food recommendations
        st.subheader("Food Recommendations")
        recommendations = recommend_food(needs, model, scaler, restrictions)
        st.table(recommendations)

if __name__ == "__main__":
    main()