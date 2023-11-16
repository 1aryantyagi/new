from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction import CustomData, PredictionPipeline


app = Flask(__name__)


@app.route('/', methods=['GET', "POST"])
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Marketing_expense=float(request.form.get('Marketing_expense')),
            Production_expense=float(request.form.get('Production_expense')),
            Multiplex_coverage=float(request.form.get('Multiplex_coverage')),
            Budget=float(request.form.get('Budget')),
            Movie_length=float(request.form.get('Movie_length')),
            Lead_Actor_Rating=float(request.form.get('Lead_Actor_Rating')),
            Lead_Actress_rating=float(request.form.get('Lead_Actress_rating')),
            Director_rating=float(request.form.get('Director_rating')),
            Producer_rating=float(request.form.get('Producer_rating')),
            Critic_rating=float(request.form.get('Critic_rating')),
            Trailer_views=int(request.form.get('Trailer_views')),
            D3_available=request.form.get('D3_available'),
            Time_taken=float(request.form.get('Time_taken')),
            Twitter_hastags=float(request.form.get('Twitter_hastags')),
            Genre=request.form.get('Genre'),
            Avg_age_actors=int(request.form.get('Avg_age_actors')),
            Num_multiplex=float(request.form.get('Num_multiplex'))
        )
        pred_df = data.get_data_as_data_frame()
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)
        result_string = f"Predicted movie Collection is: ₹{round(results[0], 2)}"
        return render_template('index.html', results=result_string)


if __name__ == "__main__":
    app.run(debug=True)
