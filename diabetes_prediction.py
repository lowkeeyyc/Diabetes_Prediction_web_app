import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Load the trained diabetes prediction model
diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))

education_levels = {
    1: "Never attended school or only kindergarten",
    2: "Grades 1 through 8 (Elementary)",
    3: "Grades 9 through 11 (Some high school)",
    4: "Grade 12 or GED (High school graduate)",
    5: "College 1 year to 3 years (Some college or technical school)",
    6: "College 4 years or more (College graduate)"
}

income_scale = {
    "Less than $10,000": 1,
    "$10,000 to less than $15,000": 2,
    "$15,000 to less than $20,000": 3,
    "$20,000 to less than $25,000": 4,
    "$25,000 to less than $35,000": 5,
    "$35,000 to less than $50,000": 6,
    "$50,000 to less than $75,000": 7,
    "$75,000 or more": 8
}

age_ranges = {
    "Age 18 - 24": 1,
    "Age 25 to 29": 2,
    "Age 30 to 34": 3,
    "Age 35 to 39": 4,
    "Age 40 to 44": 5,
    "Age 45 to 49": 6,
    "Age 50 to 54": 7,
    "Age 55 to 59": 8,
    "Age 60 to 64": 9,
    "Age 65 to 69": 10,
    "Age 70 to 74": 11,
    "Age 75 to 79": 12,
    "Age 80 or older": 13
}

# Set full-width title in the center
st.markdown(
    "<h1 style='text-align: center; color: #008080; font-size: 50px;'>Diabetes Prediction System</h1>",
    unsafe_allow_html=True,
)

health_categories = {
"Excellent": 1,
"Very Good": 2,
"Good": 3,
"Fair": 4,
"Poor": 5
}
def main():
    #st.title('Streamlit App with Right Sidebar')
    
    # Add content to the main area
    st.write("This model is trained by the responses which have been collected using direct questionnaires from the patients of Sylhet Diabetes Hospital in Sylhet, Bangladesh and approved by a doctor.")
    st.write("Refer to the info tab to calculate BMI, mental health, and physical health. If you can't find it, tap on the top-left icon '>', or refresh the page.)")
    # Add content to the sidebar
    st.sidebar.markdown('<p style="color: #008080; font-size: 28px; font-weight: bold;">Info:</p>', unsafe_allow_html=True)
    st.sidebar.markdown('**BMI:**  (Weight in kilograms) divided by (Height in meters)^2')
    st.sidebar.write("")  # Empty line for space
    st.sidebar.markdown('**Mental Health:** Stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days')
    st.sidebar.write("") 
    st.sidebar.markdown("**Physicial Health:** Physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days.")
    st.sidebar.write("") 
    st.sidebar.markdown("**NoDoctorbcCost:**  Was there a time in the past 12 months when you needed to see a doctor but could not because of cost")
    st.sidebar.write("") 

    
if __name__ == "__main__":
    main()

# Create columns for input fields
col1, col2, col3,col4 = st.columns(4)
col1.write("")
# Input fields in the first column
with col1:
    Age = st.selectbox('Age', list(age_ranges.keys()))
    Sex = st.selectbox('Gender', options=("Male", "Female"))
    HighBP = st.radio("High blood pressure", options=("Yes", "No"))
    Smoker = st.radio("Smoker", options=("Yes", "No"))
    Fruits = st.radio("Fruits",  options=("Yes", "No"))
    NoDocbcCost = st.radio("NoDoctorbcCost",  options=("Yes", "No"))


# Input fields in the second column
col2.write("")
with col2:
    BMI = st.text_input('Body mass index')
    GenHlth = st.selectbox('General Health', list(health_categories.keys()))
    HighChol = st.radio('High Cholesterol',options=("Yes", "No"))
    Stroke = st.radio('Stroke',options=("Yes", "No"))
    Veggies = st.radio('Veggies',options=("Yes", "No"))


    

# Input fields in the third column
col3.write("")
with col3:
    Education = st.selectbox('Education Level', list(education_levels.values()))
    MentHlth = st.text_input('Mental Health')
    CholCheck = st.radio('Cholesterol checkup',options=("Yes", "No"))
    HeartDiseaseorAttack = st.radio('Heart Disease or Attack',options=("Yes", "No"))
    AnyHealthcare = st.radio('AnyHealthcare',options=("Yes", "No"))
    
    
    # Other fields...
    
col4.write("")
with col4:
    Income = st.selectbox('Income Level', list(income_scale.keys()))
    PhysHlth = st.text_input('Physical Health')
    PhysActivity = st.radio('Physicial Activity',options=("Yes", "No"))
    DiffWalk = st.radio("Difficulty walking or climbing stairs",  options=("Yes", "No"))
    HvyAlcoholConsump = st.radio('Heavy Alcohol Consumption',options=("Yes", "No"))



    
    
    
    
    
diab_diagnosis = ''

if st.button("Diabetes Test Result"):
    # Convert radio button values to numeric
    HighBP_numeric = 1 if HighBP == "Yes" else 0
    Smoker_numeric = 1 if Smoker == "Yes" else 0
    Fruits_numeric = 1 if Fruits == "Yes" else 0
    NoDocbcCost_numeric = 1 if NoDocbcCost == "Yes" else 0
    DiffWalk_numeric = 1 if DiffWalk == "Yes" else 0
    
    HighChol_numeric = 1 if HighChol == "Yes" else 0
    Stroke_numeric = 1 if Stroke == "Yes" else 0
    Veggies_numeric = 1 if Veggies == "Yes" else 0
    Sex_numeric = 1 if Sex == "Male" else 0
    
    CholCheck_numeric = 1 if CholCheck == "Yes" else 0
    HeartDiseaseorAttack_numeric = 1 if HeartDiseaseorAttack == "Yes" else 0
    HvyAlcoholConsump_numeric = 1 if HvyAlcoholConsump == "Yes" else 0
    
    PhysActivity_numeric = 1 if PhysActivity == "Yes" else 0
    AnyHealthcare_numeric = 1 if AnyHealthcare == "Yes" else 0
    education_levels = {
    "Never attended school or only kindergarten": 1,
    "Grades 1 through 8 (Elementary)": 2,
    "Grades 9 through 11 (Some high school)": 3,
    "Grade 12 or GED (High school graduate)": 4,
    "College 1 year to 3 years (Some college or technical school)": 5,
    "College 4 years or more (College graduate)": 6
    }
    Education = education_levels[Education]
    
    income_scale = {
    "Less than $10,000": 1,
    "$10,000 to less than $15,000": 2,
    "$15,000 to less than $20,000": 3,
    "$20,000 to less than $25,000": 4,
    "$25,000 to less than $35,000": 5,
    "$35,000 to less than $50,000": 6,
    "$50,000 to less than $75,000": 7,
    "$75,000 or more": 8
    }
    Income = income_scale [Income]
    
    age_ranges = {
        "Age 18 - 24": 1,
        "Age 25 to 29": 2,
        "Age 30 to 34": 3,
        "Age 35 to 39": 4,
        "Age 40 to 44": 5,
        "Age 45 to 49": 6,
        "Age 50 to 54": 7,
        "Age 55 to 59": 8,
        "Age 60 to 64": 9,
        "Age 65 to 69": 10,
        "Age 70 to 74": 11,
        "Age 75 to 79": 12,
        "Age 80 or older": 13
    }
    Age = age_ranges[Age]
    
    health_categories = {
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5
    }
    GenHlth=health_categories[GenHlth]
    
    # Convert all features to float
    features = [
        float(HighBP_numeric),
        float(HighChol_numeric),
        float(CholCheck_numeric),
        float(BMI),
        float(Smoker_numeric),
        
        float(Stroke_numeric),
        float(HeartDiseaseorAttack_numeric),
        float(PhysActivity_numeric),
        float(Fruits_numeric),
        float(Veggies_numeric),
        
        float(HvyAlcoholConsump_numeric),
        float(AnyHealthcare_numeric),
        float(NoDocbcCost_numeric),
        float(GenHlth),
        float(MentHlth),
        
        float(PhysHlth),
        float(DiffWalk_numeric),
        float(Sex_numeric),
        float(Age),
        float(Education),
        float(Income)
        
    ]

    features_array = np.array(features).reshape(1, -1)

    # Load X_train from a relative path or pickle file
    #file_path = r'C:\DS_AI\projects\deploy\diabetes_prediction\X_train.csv'  # Replace this with your file path
    #data = pd.read_csv(file_path)
    #X_train = data[['HighBP', 'Smoker', 'Fruits', 'NoDocbcCost', 'DiffWalk', 'HighChol', 'Stroke', 'Veggies', 'GenHlth', 'Sex', 'CholCheck', 'HeartDiseaseorAttack', 'HvyAlcoholConsump', 'MentHlth', 'Age', 'BMI', 'PhysActivity', 'AnyHealthcare', 'PhysHlth', 'Education', 'Income']]
    # ...

    # Assuming X_train is loaded correctly
    #scaler.fit_transform(X_train)
    #scaled_features = scaler.transform(features_array)

    diab_prediction = diabetes_model.predict(features_array)

    if diab_prediction[0] == 0:
        diab_diagnosis = "The person is Not Diabetic"
    else:
        diab_diagnosis = "The person is Pre-Diabetic or Diabetic"

    # Display prediction result
    st.success(diab_diagnosis)


