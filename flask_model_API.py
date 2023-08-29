from flask import Flask, request, jsonify
import joblib

# Membuat API menggunakan FLASK
app = Flask(__name__)

# Muat model dan scaler yang telah disimpan
prediction_model = joblib.load('model_mlp.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    
    # Preprocess the input data using the loaded scaler
    scaled_data = scaler.transform([data['features']])
    
    # Perform prediction using the loaded MLP model
    prediction = prediction_model.predict(scaled_data)
    
    # Return the prediction as response
    response = {'prediction': prediction.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)