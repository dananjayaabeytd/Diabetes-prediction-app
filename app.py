from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.data
    df = pickle.loads(data)
    prediction = model.predict(df)
    return pickle.dumps({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)