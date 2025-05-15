from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('rfc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    try:
        aroma = float(request.form['aroma'])
        flavor = float(request.form['flavor'])
        aftertaste = float(request.form['aftertaste'])
        acidity = float(request.form['acidity'])
        body = float(request.form['body'])
        balance = float(request.form['balance'])
        uniformity = float(request.form['uniformity'])

        # Convert categorical values to numbers
        quakers = int(request.form['quakers'])
        color_mapping = {'Light': 0, 'Medium': 1, 'Dark': 2}
        color_encoded = int(request.form['color_encoded'])

        # Create feature array
        features = np.array([[aroma, flavor, aftertaste, acidity, body, balance, uniformity, quakers, color_encoded]])

        # Predict
        prediction = model.predict(features)[0]

        return render_template('result.html', result=prediction)

    except ValueError as e:
        return f"Invalid input: {e}", 400


if __name__ == '__main__':
    app.run(debug=True)
