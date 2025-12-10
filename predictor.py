import pandas as pd
from src.preprocess import preprocess_input  # if in root use: from preprocess import preprocess_input

def load_model():
    """
    Placeholder to load ANN model later.
    Replace with actual model loading code, e.g.:
    from tensorflow.keras.models import load_model
    return load_model('models/ann_model.h5')
    """
    return None

def predict_disease(input_dict):
    """
    input_dict: {'temperature': 34.5, 'rainfall': 12, 'humidity': 78, 'wind_speed': 10}
    Returns a placeholder string now; integrate actual model after training.
    """
    df = pd.DataFrame([input_dict])
    processed = preprocess_input(df[['temperature','rainfall','humidity','wind_speed']])
    model = load_model()
    if model is None:
        return {"prediction": "Model not added", "probability": None}
    # Example when model exists (uncomment and adapt):
    # preds = model.predict(processed)
    # return postprocess_preds(preds)
    return {"prediction": "Model not added", "probability": None}
