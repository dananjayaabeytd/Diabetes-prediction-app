from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)