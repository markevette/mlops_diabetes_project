import os
import pandas as pd
from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH
from src.data_prep import load_data, preprocess_data, save_processed

def test_load_data():
    assert os.path.exists(DATA_RAW_PATH)
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_and_save():
    df = load_data()
    df_clean = preprocess_data(df)
    save_processed(df_clean)
    assert os.path.exists(DATA_PROCESSED_PATH)
    df_processed = pd.read_csv(DATA_PROCESSED_PATH)
    assert not df_processed.empty