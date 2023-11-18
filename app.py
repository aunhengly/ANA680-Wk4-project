from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
filename = 'file_Traffic.pkl'
model = pickle.load(open(filename, 'rb'))

# Initialize an empty list to store predictions
predictions = []

@app.route('/')
def index():
    return render_template('index.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    Day_of_the_week = int(request.form['Day_of_the_week'])
    CarCount = int(request.form['CarCount'])
    BikeCount = int(request.form['BikeCount'])
    BusCount = int(request.form['BusCount'])
    TruckCount = int(request.form['TruckCount'])
    hour = int(request.form['hour'])
    minute = int(request.form['minute'])
    
    pred = model.predict(np.array([[Day_of_the_week, CarCount, BikeCount, BusCount, TruckCount, hour, minute]]))
    result = f"Day_of_the_week: {Day_of_the_week}, CarCount: {CarCount}, BikeCount: {BikeCount}, BusCount: {BusCount}, TruckCount: {TruckCount}, hour: {hour}, minute:{minute} => Prediction: {pred[0]}"
    
    # Append the result to the list of predictions
    predictions.append(result)
    
    # Display the results and clear the form
    return render_template('index.html', predict=pred[0], predictions=predictions)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
