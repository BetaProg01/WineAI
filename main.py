from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    name: str
    price: float
    tax: Optional[float] = None
    
# ORM database


app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id : int):
    return {"item_id": item_id}

@app.get("/truth")
async def root():
    return {"message": "ICC >> IA"}
