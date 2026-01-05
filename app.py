from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = 'model.pkl'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    model = None
    print("Warning: model.pkl not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                               prediction_text="Error: Model file not loaded on server.")
    
    try:
        # 1. Collect inputs
        rainfall = float(request.form['rainfall'])
        soil_quality = float(request.form['soil_quality'])
        farm_size = float(request.form['farm_size'])
        sunlight_hours = float(request.form['sunlight_hours'])
        fertilizer = float(request.form['fertilizer'])

        # 2. Biological Validation Logic
        # If there is no rain or no farm size, the yield MUST be zero
        if rainfall <= 0 or farm_size <= 0:
            output = 0.0
            note = " (Note: Zero yield due to lack of rainfall or farm size)"
        else:
            # 3. Format for XGBoost
            input_features = [rainfall, soil_quality, farm_size, sunlight_hours, fertilizer]
            final_features = [np.array(input_features)]
            
            # 4. Predict
            prediction = model.predict(final_features)
            # Ensure the model never returns a negative number by using max(0, ...)
            output = round(max(0.0, float(prediction[0])), 2)
            note = ""

        return render_template('index.html', 
                               prediction_text=f'Estimated Crop Yield: {output} kg{note}')
    
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
