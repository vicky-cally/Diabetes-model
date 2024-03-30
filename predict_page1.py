import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('classifier_knn.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model() 
  
model = data['model']

def show_predict_page():
    # Title with custom styling
    st.markdown(
        f"<h1 style='text-align: center; color: purple;'>Diabetes Prediction Tool</h1>",
        unsafe_allow_html=True
    )

    # Write text with custom styling
    st.markdown(
        f"""
        <div style='text-align: justify; color: #2c3e50;'>
            <h2 style='color: purple;'>About the Tool:</h2>
            This tool utilizes machine learning to predict the likelihood of an individual having diabetes based on their health indicators.
            Input your health metrics below, and our model will provide a prediction of whether you are classified as diabetic or non-diabetic.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Input variables section
    st.markdown(
        f"""
        <div style='text-align: justify; color: #2c3e50;'>
            <h2 style='color: purple;'>Input Variables:</h2>
            <ul>
                <li>Pregnancies: Number of pregnancies the individual has had.</li>
                <li>Glucose: Blood sugar level.</li>
                <li>Blood Pressure: Systolic blood pressure measurement.</li>
                <li>Skin Thickness: Thickness of skin folds (triceps skinfold thickness).</li>
                <li>Insulin: Insulin level.</li>
                <li>BMI: Body mass index.</li>
                <li>Diabetes Pedigree Function: Diabetes pedigree function score.</li>
                <li>Age: Age of the individual.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Output section
    st.markdown(
        f"""
        <div style='text-align: justify; color: #2c3e50;'>
            <h2 style='color: purple;'>Output:</h2>
            <p>Diabetic or Non-Diabetic: Prediction of whether the individual is classified as having diabetes or not.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Instructions section
    st.markdown(
        f"""
        <div style='text-align: justify; color: #2c3e50;'>
            <h2 style='color: purple;'>Instructions:</h2>
            <ol>
                <li>Enter the required health metrics in the input fields provided.</li>
                <li>Click the "Predict" button to generate the prediction.</li>
                <li>The result will be displayed indicating whether the individual is classified as diabetic or non-diabetic.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Disclaimer section
    st.markdown(
        f"""
        <div style='text-align: justify; color: #2c3e50;'>
            <h2 style='color: purple;'>Disclaimer:</h2>
            <p>This tool is intended for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
            Please consult with a healthcare professional for any medical concerns or decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add input fields for each feature
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17)
    glucose = st.number_input("Glucose", min_value=44, max_value=199)
    blood_pressure = st.number_input("Blood Pressure", min_value=24, max_value=122)
    skin_thickness = st.number_input("Skin Thickness", min_value=7, max_value=99)
    insulin = st.number_input("Insulin", min_value=14, max_value=846)
    bmi = st.number_input("BMI", min_value=18.2, max_value=67.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42)
    age = st.number_input("Age", min_value=21,max_value=81)

    # Add a button to trigger prediction
    if st.button("Predict"):
        # Prepare input data as a numpy array
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        # Use the loaded model to make predictions
        prediction = model.predict(features)
        # Display the prediction
        st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")

# Call the function to show the prediction page
#show_predict_page()
