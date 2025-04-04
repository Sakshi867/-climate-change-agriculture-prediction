import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset (replace with your dataset's actual path)
df = pd.read_csv(r'C:\Users\mgmce\Downloads\climate_change_impact_on_agriculture_2024 (1).csv')

# Preprocessing: Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Define features (X) and target (y)
X = df[['Average_Temperature_C', 'Total_Precipitation_mm', 'Soil_Health_Index']]  # Features
y = df['Crop_Yield_MT_per_HA']  # Target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('crop_yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'crop_yield_model.pkl'")
