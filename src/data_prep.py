import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH, TEST_SIZE, RANDOM_STATE

def load_data():
    """Load raw dataset from CSV."""
    df = pd.read_csv(DATA_RAW_PATH)
    return df

def preprocess_data(df):
    """Basic preprocessing: handle missing values."""
    df = df.dropna()  # simple cleaning for now
    return df

def split_data(df):
    """Split into train and test sets."""
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def save_processed(df):
    """Save cleaned dataset."""
    df.to_csv(DATA_PROCESSED_PATH, index=False)

if __name__ == "__main__":
    df = load_data()
    df_clean = preprocess_data(df)
    save_processed(df_clean)
    print("Data preprocessing complete.")