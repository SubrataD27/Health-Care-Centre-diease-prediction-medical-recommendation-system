from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process

app = Flask(__name__)

# Load datasets from Kaggle
sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
workout = pd.read_csv("kaggle_dataset/workout_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")
medications = pd.read_csv("kaggle_dataset/medications.csv")
diets = pd.read_csv("kaggle_dataset/diets.csv")

# Load the Random Forest model
Rf = pickle.load(open('model/RandomForest.pkl', 'rb'))

# Dictionary of symptoms and diseases
symptoms_list = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    # Add more symptoms as needed...
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    # Add more diseases as needed...
}

# Preprocess symptoms list for easier matching
symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}

# Function to retrieve information about the disease
def information(predicted_dis):
    disease_description = description.loc[description['Disease'] == predicted_dis, 'Description'].values
    disease_precautions = precautions.loc[precautions['Disease'] == predicted_dis, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    disease_medications = medications.loc[medications['Disease'] == predicted_dis, 'Medication'].values
    disease_diet = diets.loc[diets['Disease'] == predicted_dis, 'Diet'].values
    disease_workout = workout.loc[workout['disease'] == predicted_dis, 'workout'].values

    # Handle missing values
    disease_description = disease_description[0] if len(disease_description) > 0 else "No description available."
    disease_precautions = disease_precautions[0].tolist() if len(disease_precautions) > 0 else ["No precautions available."]
    disease_medications = disease_medications[0].split(",") if len(disease_medications) > 0 else ["No medications available."]
    disease_diet = disease_diet[0].split(",") if len(disease_diet) > 0 else ["No diet recommendations available."]
    disease_workout = disease_workout[0].split(",") if len(disease_workout) > 0 else ["No workout suggestions available."]

    return disease_description, disease_precautions, disease_medications, disease_diet, disease_workout

# Function to predict the disease based on symptoms
def predicted_value(patient_symptoms):
    # Ensure i_vector has the same size as the model's expected input
    expected_features = Rf.n_features_in_  # Get the number of features the model expects
    i_vector = np.zeros(expected_features)

    # Map patient symptoms to the vector
    for symptom in patient_symptoms:
        if symptom in symptoms_list_processed:
            i_vector[symptoms_list_processed[symptom]] = 1

    try:
        # Predict the disease
        predicted_label = Rf.predict([i_vector])[0]
        return diseases_list.get(predicted_label, "Unknown Disease")
    except ValueError as e:
        # Handle mismatch errors
        raise ValueError(f"Input feature vector has a mismatch: {str(e)}")

# Function to correct spelling of symptoms using fuzzy matching
def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    return closest_match if score >= 80 else None

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms_input = request.form.get('symptoms', '').strip()
        if not symptoms_input:
            message = "Please enter symptoms for prediction."
            return render_template('index.html', message=message)

        # Split and clean the user's input
        patient_symptoms = [s.strip().lower() for s in symptoms_input.split(',')]

        # Correct spelling and validate symptoms
        corrected_symptoms = []
        for symptom in patient_symptoms:
            corrected_symptom = correct_spelling(symptom)
            if corrected_symptom:
                corrected_symptoms.append(corrected_symptom)
            else:
                message = f"Symptom '{symptom}' not found in the database."
                return render_template('index.html', message=message)

        # Predict the disease
        predicted_disease = predicted_value(corrected_symptoms)

        # Retrieve additional information about the disease
        dis_des, precautions, medications, rec_diet, workout = information(predicted_disease)

        return render_template(
            'index.html',
            symptoms=corrected_symptoms,
            predicted_disease=predicted_disease,
            dis_des=dis_des,
            my_precautions=precautions,
            medications=medications,
            my_diet=rec_diet,
            workout=workout
        )

    return render_template('index.html')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
