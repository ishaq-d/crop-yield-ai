import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# 1. Load your dataset
# Make sure 'crop_yield_data.csv' is inside your web_app folder!
crop_dataset = pd.read_csv('crop_yield_data.csv')

# 2. Define Features and Target
# Make sure these column names match your CSV exactly
X = crop_dataset[['rainfall', 'soil_quality', 'farm_size', 'sunlight_hours', 'fertilizer']]
y = crop_dataset['crop_yield']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the XGBoost Model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Save the brain
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("XGBoost Model saved successfully as model.pkl")