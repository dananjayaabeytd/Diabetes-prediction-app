from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import pandas as pd

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Temporarily allow all origins for testing

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)