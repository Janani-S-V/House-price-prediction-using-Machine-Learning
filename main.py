# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

# Load the dataset (assuming you have a dataset named 'house_data.csv')
df = pd.read_csv('train.csv')

# Select features and target variable
X = df[['number of bedrooms', 'number of bathrooms','number of floors','Postal Code']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from HTML form
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return render_template('index.html', prediction_text='Predicted Price: INR {:,.2f}'.format(prediction[0]))
    

if __name__ == "__main__":
    app.run(debug=True)
