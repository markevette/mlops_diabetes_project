import os
from src.config import MODEL_PATH
from src.train import load_processed_data, train_model, evaluate_model
from src.data_prep import split_data

def test_training_pipeline():
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)

    assert acc >= 0.0  # just sanity check
    assert acc <= 1.0

def test_model_saved():
    # run training once before this test if needed
    assert os.path.exists(MODEL_PATH)