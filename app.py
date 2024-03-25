from flask import Flask, request, render_template
import numpy as np
import pandas as pd
# Ensure the import paths are correct based on your project structure
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])  # Make sure methods are properly defined
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            gender=request.form['gender'],
            race_ethnicity=request.form['ethnicity'],  # Ensure form field names match
            parental_level_of_education=request.form['parental_level_of_education'],
            lunch=request.form['lunch'],
            test_preparation_course=request.form['test_preparation_course'],
            reading_score=float(request.form['reading_score']),  # Corrected form field names
            writing_score=float(request.form['writing_score'])
        )
        pred_df = data.get_data_as_dataframe()  # Ensure this method name matches the actual method
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    return render_template('home.html')  # Handles GET request by showing the form

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
