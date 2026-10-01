from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["session_duration", "click_count", "cart_additions", "time_of_day"]
WEEKDAY_MAP = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

@dataclass
class DatasetSplit:
    """Dataclass to hold train and test splits."""
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    timestamps: pd.Series = None

class DatasetProcessor:
    """Handles raw data ingestion and feature engineering."""
    
    def load_events(self, data_path):
        print(f"Loading data from {data_path}...")
        return pd.read_csv("diamonds (1).csv")

    def create_session_features(self, events):
        print("Creating session features...")
        # Placeholder for actual feature engineering logic
        return events

    def prepare_features(self, session_features):
        print("Preparing features and target variables...")
        # Ensure your dataset contains these columns or adjust FEATURE_COLUMNS
        X = session_features[FEATURE_COLUMNS]
        y = session_features["is_order"] # Target column (0 or 1)
        timestamps = session_features.get("timestamp", pd.Series([None]*len(X)))
        return X, y, timestamps

    def split_data(self, x, y, timestamps):
        print("Splitting dataset into train and test...")
        # Chronological split recommended for session data (shuffle=False)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=False
        )
        return DatasetSplit(x_train, x_test, y_train, y_test, timestamps)