from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict

app = FastAPI(title="Diabetes Prediction API")

model = load_model()

class InputData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict_diabetes(data: InputData):
    input_dict = data.dict()
    result = predict(model, input_dict)
    return {"prediction": result}