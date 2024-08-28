from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn

model = joblib.load("logreg_model.joblib")

class CustomerData(BaseModel) :
    tenure : int
    InternetService : str
    OnlineSecurity : str
    TechSupport : str
    Contract : str
    PaymentMethod : str

app = FastAPI()

@app.post("/predict")
def predict(data : CustomerData):
    input_data = {
        "tenure" : data.tenure,
        'InternetService' : data.InternetService,
        'OnlineSecurity' : data.OnlineSecurity,
        'TechSupport' : data.TechSupport,
        'Contract' : data.Contract,
        'PaymentMethod' : data.PaymentMethod
    }

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)

    return {"prediction" : int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# input eg - {72, Fiber optic, Yes, Yes, Two year, Credit card (automatic)}
# uvicorn app:app --reload
#  curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d2, \"InternetService\": \"Fiber optic\", \"OnlineSecurity\": \"Yes\", \"TechSupport\": \"Yes\", \"Contract\": \"Two year\", \"PaymentMethod\": \"Credit card (automatic)\"}"