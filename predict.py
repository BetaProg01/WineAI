import numpy as np
import joblib


def predict(data_to_predict):
    loaded_model = joblib.load('wine_quality_model.joblib')
    prediction = loaded_model.predict(new_data)
    prediction = np.round(prediction).astype(int)
    
    return prediction


