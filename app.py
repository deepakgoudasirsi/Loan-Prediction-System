from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and feature names
model = joblib.load('loan_prediction_model.joblib')
feature_names = joblib.load('feature_names.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_term']),
            'Credit_History': float(request.form['credit_history']),
            'Property_Area': request.form['property_area']
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the data
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                             'Self_Employed', 'Property_Area']
        
        for column in categorical_columns:
            df[column] = df[column].map(lambda x: 1 if x == 'Yes' else 0)
        
        # Make prediction
        prediction = model.predict(df[feature_names])[0]
        probability = model.predict_proba(df[feature_names])[0]
        
        result = {
            'prediction': 'Approved' if prediction == 1 else 'Not Approved',
            'probability': float(max(probability))
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 