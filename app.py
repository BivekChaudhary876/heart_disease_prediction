from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extracting features from the JSON
    features = np.array([[
        data['BMI'],
        data['Smoking'],
        data['Alcohol_drinking'],
        data['Stroke'],
        data['Physical_health'],
        data['Mental_health'],
        data['Diff_walking'],
        data['Sex'],
        data['Age_category'],
        data['Race'],
        data['Diabetic'],
        data['Physical_activity'],
        data['General_health'],
        data['Sleep_time'],
        data['Asthma'],
        data['Kidney_disease'],
        data['Skin_cancer']
    ]])

    # Make prediction
    heart_disease_probability = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]
    # Convert to JSON serializable format
    result = {
        'heart_disease_probability': float(heart_disease_probability),
        'prediction': 'Disease' if prediction == 1 else 'No Disease'
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
