import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_input(df):
    """
    Preprocess weather input data.
    Expected numeric columns: temperature, rainfall, humidity, wind_speed
    Returns scaled DataFrame.
    """
    scaler = MinMaxScaler()
    # Fit-transform expects numeric values only
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    return scaled_df
