import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model_with_lstm-input(4)(1).h5')
scaler = pickle.load(open('scaler-input(4).pkl','rb'))
scaler_input = pickle.load(open('X_scaler-input(4).pkl','rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    # Get input data from request
    int_features = [float(x) for x in request.form.values()]  # Extract input features from request

    # Scale the input features
    scaled_features = scaler_input.transform([int_features])
    final_features = np.array(scaled_features).reshape(1, 1, len(int_features))  # Reshape input data to match model's input shape
    
    # Make prediction using the loaded model
    prediction_scaled = model.predict(final_features)
    
    # Inverse transform the prediction to get the actual predicted value
    predicted_value = scaler.inverse_transform(prediction_scaled)[0,0]

    # Return the prediction as JSON response
    return render_template('home.html', prediction_text="Walmart Sales {:.2f}".format(predicted_value))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)