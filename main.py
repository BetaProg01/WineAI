from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from model import *
from predict import predict


app = FastAPI()


@app.get("/api/predict") # envoie les statistiques d'un vin supposé parfait (qui maximise la qualité)
async def perfect_wine(): # TODO
    return {}

@app.get("/api/model/description") # envoie un ensemble d'infos décrivant qualitativement le modèle (type, métriques, ...)
async def give_model_description(): # TODO
    return {}

@app.post("/api/predict") # prend en entrée les données du vin et renvoie sa prédiction de qualité sur 10
async def predict_wine(data: WineData):
    return predict(data)

@app.get("/api/model") # envoie un fichier contenant la sérialisation du modèle actuel
async def give_model():
    # Replace 'path/to/your/file.txt' with the actual path to your file
    file_path = './wine_quality_model.joblib'
    
    # Specify the filename to be used when downloading the file
    filename = 'wine_quality_model.joblib'
    
    # Return the file using FileResponse
    return FileResponse(file_path, filename=filename)

@app.put("/api/model") # prend un json décrivant un nouveau vin à ajouter dans la base d'entraînement
async def enrich_base(new_data: NewWineData):
    try:
        # Read the existing CSV file
        csv = pd.read_csv("Wines.csv")

        # Add the new data as a new row
        correct_row = dict()
        correct_row["Id"] = int(len(csv))
        correct_row["fixed acidity"] = new_data.fixed_acidity
        correct_row["volatile acidity"] = new_data.volatile_acidity
        correct_row["citric acid"] = new_data.citric_acid
        correct_row["residual sugar"] = new_data.residual_sugar
        correct_row["chlorides"] = new_data.chlorides
        correct_row["free sulfur dioxide"] = new_data.free_sulfur_dioxide
        correct_row["total sulfur dioxide"] = new_data.total_sulfur_dioxide
        correct_row["density"] = new_data.density
        correct_row["pH"] = new_data.pH
        correct_row["sulphates"] = new_data.sulphates
        correct_row["alcohol"] = new_data.alcohol
        correct_row["quality"] = new_data.quality
        csv = pd.concat([csv, pd.DataFrame([correct_row])], ignore_index=True)

        # Save the updated CSV file
        csv.to_csv("Wines.csv", index=False)

        res = "Data added to the training set"
    except Exception as e:
        # Log the exception for debugging purposes
        res = "Error while adding data to the training set"
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    return {"Result": res}

@app.post("/api/model/retrain") # réentraîne le modèle (avec les nouvelles données du put), et écrase la sérialisation du dernier
async def retrain():
    createModel()
    return{}

@app.get("/truth")
async def root():
    return {"message": "ICC >> IA >> CYBER"}
