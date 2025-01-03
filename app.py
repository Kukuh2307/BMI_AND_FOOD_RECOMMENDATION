import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to prepare and train the model
# def train_food_recommendation_model():
#     try:
#         # Load and prepare dataset
#         df = pd.read_csv("combined_nutrition_dataset.csv")
        
#         # Define features
#         features = ['Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g', 'Fiber_g', 'VitC_mg', 'Calcium_mg']
        
#         # Handle missing values
#         df[features] = df[features].fillna(df[features].mean())
        
#         # Create feature matrix X
#         X = df[features].copy()  # Explicitly create X
        
#         # Create nutritional score
#         df['nutritional_score'] = (
#             df['Protein_g'] / df['Energy_kcal'].replace(0, 1) * 100 +
#             df['Fiber_g'] * 2 +
#             df['VitC_mg'] / 60 +
#             df['Calcium_mg'] / 1000
#         )
        
#         # Handle any NaN in nutritional score
#         df['nutritional_score'] = df['nutritional_score'].fillna(0)
        
#         # Create categories and target variable y
#         df['category'] = pd.qcut(df['nutritional_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
#         y = df['category']
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Scale features
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         # Train model
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X_train_scaled, y_train)
        
#         # Save model and scaler
#         with open("food_recommendation_model.pkl", "wb") as file:
#             pickle.dump((model, scaler), file)
        
#         return model, scaler
        
#     except Exception as e:
#         st.error(f"Error in training model: {str(e)}")
#         print(f"Error details: {e}")
#         return None, None

# Function to calculate BMI and nutritional needs
def calculate_bmi_and_needs(weight, height, age, gender, activity_level):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    # Determine BMI category
    if bmi < 18.5:
        bmi_category = 'Underweight'
    elif 18.5 <= bmi < 25:
        bmi_category = 'Normal'
    elif 25 <= bmi < 30:
        bmi_category = 'Overweight'    
    else:
        bmi_category = 'Obese'
    
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
        'BMI': round(bmi, 2),
        'Category': bmi_category,
        'Daily Calories (kcal)': round(daily_calories),
        'Protein (g)': round(protein_needs),
        'Fat (g)': round(fat_needs),
        'Carbohydrate (g)': round(carb_needs)
    }

# Function to recommend food based on nutritional needs
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


# Main Streamlit UI
def main():
    st.title("BMI Calculator and Food Recommendation System")
    
    # Try to load the model, train if not exists
    try:
        with open("food_recommendation_model.pkl", "rb") as file:
            model, scaler = pickle.load(file)
    except FileNotFoundError:
        st.info("Training model for first use...")
        # model, scaler = train_food_recommendation_model()
        st.success("Model training completed!")
    
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