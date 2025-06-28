import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Load your trained pipeline
model_pipeline = joblib.load('creditworthiness_pipeline.pkl')

# Create the Flask app
app = Flask(__name__)

# Home route – renders the form
@app.route('/')
def home():
    return render_template('index.html')  # Make sure this file exists in /templates

# Predict route – handles form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read data from the form
        input_data = {
            'city': request.form['city'],
            'sex': request.form['sex'],
            'social_class': request.form['social_class'],
            'primary_business': request.form['primary_business'],
            'age': int(request.form['age']),
            'annual_income': float(request.form['annual_income']),
            'monthly_expenses': float(request.form['monthly_expenses']),
            'loan_tenure': int(request.form['loan_tenure']),
            'loan_amount': float(request.form['loan_amount']),
            'DTI': float(request.form['dti']),
            'Disposable_Income': float(request.form['disposable_income']),
            'LTV': float(request.form['ltv']),
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict using the pipeline
        prediction = model_pipeline.predict(input_df)[0]
        result = "✅ Creditworthy" if prediction == 1 else "❌ Not Creditworthy"

    except Exception as e:
        result = f"Error during prediction: {e}"

    # Show result on the same page
    return render_template('index.html', prediction_text=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
