from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
# Enable CORS on the app
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)