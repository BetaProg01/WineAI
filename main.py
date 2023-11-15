from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    name: str
    price: float
    tax: Optional[float] = None
    
# ORM database


app = FastAPI()

@app.get("/truth")
async def root():
    return {"message": "ICC >> IA >> CYBERSEX"}

@app.post("/api/predict") # prend en entrée les données du vin et renvoie sa prédiction de qualité sur 10
async def predict_wine():
    return {}

@app.get("/api/predict") # envoie les statistiques d'un vin supposé parfait (qui maximise la qualité)
async def perfect_wine():
    return {}

@app.get("/api/model") # envoie un fichier contenant la sérialisation du modèle actuel
async def give_model():
    return {}

@app.get("/api/model/description") # envoie un ensemble d'infos décrivant qualitativement le modèle (type, métriques, ...)
async def give_model_description():
    return {}

@app.put("/api/model") # prend un json décrivant un nouveau vin à ajouter dans la base d'entraînement
async def enrich_base():
    return {}

@app.post("/api/model/retrain") # réentraîne le modèle (avec les nouvelles données du put), et écrase la sérialisation du dernier
async def retrain():
    return{}
