import joblib
import pandas as pd
from src.config import MODEL_PATH

def load_model():
    """Load the trained model."""
    return joblib.load(MODEL_PATH)

def predict(model, input_data: dict):
    """Run prediction on a single input sample."""
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return int(prediction)