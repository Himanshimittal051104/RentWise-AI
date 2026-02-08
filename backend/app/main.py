from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model + columns
model = joblib.load("model/house_model.pkl")
columns = joblib.load("model/columns.pkl")

@app.get("/")
def home():
    return {"message": "Rent Prediction API running"}

@app.post("/predict")
def predict(data: dict):

    # numeric inputs
    input_data = {
        "BHK": data["BHK"],
        "Size": data["Size"],
        "Bathroom": data["Bathroom"]
    }

    df = pd.DataFrame([input_data])

    # categorical columns
    city_col = "City_" + data["City"]
    area_col = "Area Locality_" + data["Area_Locality"]

    df[city_col] = 1
    df[area_col] = 1

    # fill missing columns with 0
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    prediction = model.predict(df)
    final_prediction = max(0, prediction[0])

    return {"predicted_rent": float(final_prediction)}

