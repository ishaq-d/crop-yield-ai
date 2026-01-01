from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model with a safety check for the web server
model_path = 'model.pkl'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    model = None
    print("Warning: model.pkl not found. Make sure to run your training script!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                               prediction_text="Error: Model file not loaded on server.")
    
    try:
        # 1. Collect inputs exactly as you did locally
        input_features = [
            float(request.form['rainfall']),
            float(request.form['soil_quality']),
            float(request.form['farm_size']),
            float(request.form['sunlight_hours']),
            float(request.form['fertilizer'])
        ]
        
        # 2. Format for XGBoost
        final_features = [np.array(input_features)]
        
        # 3. Predict
        prediction = model.predict(final_features)
        
        # 4. Result
        output = round(float(prediction[0]), 2)

        return render_template('index.html', 
                               prediction_text=f'Estimated Crop Yield: {output} kg')
    
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    # Render uses the PORT environment variable, so we add this
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)