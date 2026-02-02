import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.config import DATA_PROCESSED_PATH, MODEL_PATH, TEST_SIZE, RANDOM_STATE
from src.data_prep import split_data

def load_processed_data():
    """Load processed dataset."""
    return pd.read_csv(DATA_PROCESSED_PATH)

def train_model(X_train, y_train):
    """Train a simple logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy."""
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc

def save_model(model):
    """Save trained model."""
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    save_model(model)

    print(f"Training complete. Accuracy: {accuracy:.4f}")