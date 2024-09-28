from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
with open('./model/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     df = pd.DataFrame(data)
#     prediction = model.predict(df)
#     return jsonify({'prediction': prediction.tolist()})

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Perform prediction logic here
    prediction = [0]  # Example prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)