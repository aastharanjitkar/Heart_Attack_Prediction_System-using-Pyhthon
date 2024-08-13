from flask import Flask, request, render_template
from model import LogisticRegression, load_data
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        age = float(request.form['age'])
        gender = float(request.form['gender'])
        impulse = float(request.form['impluse'])
        pressure_high = float(request.form.get('pressurehigh', 0))
        pressure_low = float(request.form['pressurelow'])
        glucose = float(request.form['glucose'])
        kcm = float(request.form['kcm'])
        troponin = float(request.form['troponin'])

        input_data = [[age, gender, impulse, pressure_high, pressure_low, glucose, kcm, troponin]]
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)

        result = "Low risk of Heart Attack" if prediction[0] == 0 else "High risk of Heart Attack"

        return render_template('predict.html', result=result, impulse=impulse, pressurehigh=pressure_high, pressurelow=pressure_low, glucose=glucose, troponin=troponin)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
