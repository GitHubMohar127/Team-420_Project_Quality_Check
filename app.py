# from flask import Flask, render_template, request
# import pandas as pd
# import sklearn
# import pickle

# app=Flask(__name__)

# model = pickle.load(open("random_forest_model.pkl", 'rb'))
# med = pd.read_csv("medicine_quality_dataset.csv")

# @app.route('/')
# def index():
#     Batch_Number = med['Batch Number'].unique()

#     return render_template('ind.html', Batch_Number=Batch_Number)


# @app.route('/predict', methods=['POST'])
# def predict():

#     batch_number = request.form.get('batch_number')
#     print(batch_number)


#     prediction = model.predict(pd.DataFrame([[batch_number]], columns=['batch_number']))
#     print(prediction)
#     return ""


# if __name__ == "__main__":
#     app.run(debug=True)







from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import datetime

app = Flask(__name__)

# Load the pre-trained RandomForest model (Ensure it's compatible with your current scikit-learn version)
model = pickle.load(open("random_forest_model.pkl", 'rb'))

# Load and preprocess the medicine dataset
med = pd.read_csv("medicine_quality_dataset.csv")

# Convert 'Expiration Date' and 'Manufacture Date' to datetime format
med['Expiration Date'] = pd.to_datetime(med['Expiration Date'])
med['Manufacture Date'] = pd.to_datetime(med['Manufacture Date'])

# Home page route (render HTML template with available batch numbers)
@app.route('/')
def index():
    batch_numbers = med['Batch Number'].unique()  # Get unique batch numbers for the dropdown
    return render_template('ind.html', Batch_Number=batch_numbers)

# Prediction endpoint (handles AJAX request)
@app.route('/predict', methods=['POST'])
def predict():
    batch_number = request.form.get('batch_number')  # Get the selected batch number from the form

    # Filter the dataset to get the corresponding batch data
    batch_data = med[med['Batch Number'] == batch_number]

    if batch_data.empty:
        return jsonify({'error': 'Batch number not found'})

    # Extract features required by the model
    features = batch_data[['Purity Level', 'Storage Conditions', 'Appearance', 'Dissolution Rate']].copy()
    features['Days to Expiry'] = (batch_data['Expiration Date'] - pd.Timestamp.now()).dt.days

    # Make prediction using the loaded model
    predicted_quality = model.predict(features)

    # Construct the response
    result = {
        'Medicine Name': batch_data['Medicine Name'].values[0],
        'Expiration Date': batch_data['Expiration Date'].values[0].strftime('%Y-%m-%d'),
        'Manufacture Date': batch_data['Manufacture Date'].values[0].strftime('%Y-%m-%d'),
        'Dissolution Rate': batch_data['Dissolution Rate'].values[0],
        'Quality Check': predicted_quality[0]
    }
    
    # Send result back to the frontend as a JSON response
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
