import io
import os
import torch
import numpy as np
import torch.nn as nn
from fastapi import Request
from dotenv import load_dotenv
from pymongo.server_api import ServerApi
from fastapi.staticfiles import StaticFiles
from backend.model.asr_model import ASRModel
from pymongo.mongo_client import MongoClient
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from backend.routes.asr_routes import asr_router
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

app = FastAPI()

'''
# MongoDB Connection
mongo_uri = os.getenv("MONGODB_URI")
if not mongo_uri:
    raise ValueError("MongoDB URI not found. Please check the .env file.")
'''

uri="mongodb+srv://azharhuzaifa123:1imagination@wastes.ew1kl.mongodb.net/?retryWrites=true&w=majority&appName=Wastes"

# Create a new client and connect to the server
client = AsyncIOMotorClient(uri, server_api=ServerApi('1'))
db = client["Wastes"]

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# MongoDB Connection
#client = MongoClient(mongo_uri, server_api=ServerApi('1'))
#client = AsyncIOMotorClient(uri)


'''
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

'''


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
#client = AsyncIOMotorClient("mongodb://localhost:27017")

# Include your ASR routes
app.include_router(asr_router)

templates = Jinja2Templates(directory="frontend/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define model loading and prediction logic
class ASRModelWrapper:
    def __init__(self, model_path: str):
        self.model = ASRModel(input_dim=13, hidden_dim=128, vocab_size=35)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, np_array):
        inputs = torch.tensor(np_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(inputs)
            predicted_indices = torch.argmax(outputs, dim=-1).squeeze().tolist()
            return predicted_indices  # Convert indices to actual labels if needed

# Initialize model (make sure to replace with actual model path)
asr_model = ASRModelWrapper("G://Automatic Speecch Recognition System//model.pth")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_content = await file.read()
    file_type = file.content_type

    if file_type == "application/x-npy":
        np_array = np.load(io.BytesIO(file_content))
    else:
        return {"error": "Unsupported file format"}

    # Pass np_array to the model for prediction
    predictions = asr_model.predict(np_array)
    prediction_str = ''.join([chr(idx) for idx in predictions])  # Assuming prediction returns character indices

    return {"prediction": prediction_str}

