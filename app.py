#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_salary():
    # Load the model and scaler
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Get the input data from the request
    input_data = request.get_json()

    # Convert the input data into a dataframe
    input_df = pd.DataFrame({'YearsExperience': [input_data['yearsOfExperience']]})

    # Scale the input data using the scaler object
    input_scaled = scaler.transform(input_df)

    # Make the prediction using the model
    prediction = model.predict(input_scaled)

    # Convert the prediction to a string and return the response
    return jsonify({'salary': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




