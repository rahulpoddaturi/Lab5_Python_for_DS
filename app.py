import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle


app = Flask(__name__)
## open and load the pickle file provided in read mode.
model = pickle.load(open('predict_car.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age_of_the_car = int(request.form['Age_Of_The_Car'])
    kms_driven = int(request.form['Kms_Driven'])
    transmission = int(request.form['Transmission'])
    fuel_type = int(request.form['Fuel_Type'])
    owner = int(request.form['Owner'])
    seller_type = int(request.form['Seller_Type'])
    present_price = float(request.form['Present_Price'])
    prediction = model.predict([[present_price,kms_driven,fuel_type,seller_type,transmission,owner,age_of_the_car]])

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price of the car is {} Lakhs'.format(output))


if __name__ == "__main__":
    app.run(debug=True , port = 5006)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
