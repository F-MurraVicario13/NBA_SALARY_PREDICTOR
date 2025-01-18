from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('nba_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('nba_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Please run the Jupyter notebook first to train and save the model.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        input_data = {
            'Age': float(request.form['age']),
            'MP_per_game': float(request.form['minutes']),
            'PTS_per_game': float(request.form['points']),
            'AST_per_game': float(request.form['assists']),
            'TRB_per_game': float(request.form['rebounds']),
            'STL_per_game': float(request.form['steals']),
            'BLK_per_game': float(request.form['blocks']),
            'FG%_per_game': float(request.form['fg_percentage']) / 100,
            '3P%_per_game': float(request.form['three_percentage']) / 100,
            'FT%_per_game': float(request.form['ft_percentage']) / 100,
            'PER_advanced': float(request.form['per']),
            'WS_advanced': float(request.form['win_shares'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            'success': True,
            'prediction': f'${prediction:,.2f}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)