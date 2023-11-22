import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# WineData is used to receive a description of a wine from the user and send its predicted quality
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# NewWineData is used to add new rows to the training dataset from users
class NewWineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality: int

# Description of the model to send to the user
class ModelDescription(BaseModel):
    type: str
    metrics: str
    short_desc: str

# seed for random
SEED = 4269

def createModel():
    # Importing the dataset
    Vins = pd.read_csv("Wines.csv")

    # Upsample some datas that are not enough represented 
    class_3 = Vins[Vins.quality==3]           
    class_4 = Vins[Vins.quality==4]          
    class_5 = Vins[Vins.quality==5]   
    class_6 = Vins[Vins.quality==6] 
    class_7 = Vins[Vins.quality==7]     
    class_8 = Vins[Vins.quality==8]    

    class_3_upsampled = resample(class_3, replace=True, n_samples=600, random_state=SEED) 
    class_4_upsampled = resample(class_4, replace=True, n_samples=600, random_state=SEED) 
    class_7_upsampled = resample(class_7, replace=True, n_samples=600, random_state=SEED) 
    class_8_upsampled = resample(class_8, replace=True, n_samples=600, random_state=SEED) 

    Balanced_data = pd.concat([class_3_upsampled, class_4_upsampled, class_7_upsampled, class_8_upsampled, class_5, class_6]).reset_index(drop=True)


    # Prepare train and test datasets
    X = Balanced_data.drop(columns=['quality', 'Id'])
    y = Balanced_data['quality']

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Creation and training of the model
    model = RandomForestRegressor(n_estimators=500, random_state=SEED)

    model.fit(X_train, y_train)


    # Evaluations 
    predictions = model.predict(X_test)
    rounded_predictions = np.round(predictions)

    rounded_predictions = rounded_predictions.astype(int)
    y_test_classes = y_test.astype(int)

    conf_matrix = confusion_matrix(y_test_classes, rounded_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    mse = mean_squared_error(y_test, predictions)
    print(f'Initial Mean Squared Error: {mse}')

    accuracy = accuracy_score(y_test_classes, rounded_predictions)
    print("Accuracy:", accuracy)


    # Saving model
    joblib.dump(model, 'wine_quality_model.joblib')