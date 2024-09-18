import torch
import librosa
import numpy as np
from backend.phonetics import ARPAbet
from backend.model.asr_model import load_asr_model
from motor.motor_asyncio import AsyncIOMotorClient

model = load_asr_model()

async def process_audio(audio_file):
    # Load the audio file
    audio_data, sr = librosa.load(audio_file.file, sr=16000)
    
    # Extract MFCC features (adjust this as per your MFCC extraction)
    mfcc_features = librosa.feature.mfcc(audio_data, sr=sr, n_mfcc=13)
    mfcc_features = torch.tensor(mfcc_features).unsqueeze(0)
    
    # Predict the transcript using the ASR model
    predicted_output = model.predict(mfcc_features)
    
    # Convert the output to text
    transcript = decode_output(predicted_output)
    
    # Save the transcript to MongoDB
    await save_transcript_to_db(audio_file.filename, transcript)
    
    return transcript

def decode_output(model_output):
    """Decodes the output of the ASR model into human-readable text."""
    # Assuming model_output is a tensor of shape (batch_size, seq_len, vocab_size)
    predicted_indices = torch.argmax(model_output, dim=-1)  # Get the most probable index
    
    # Convert indices to phonemes using the ARPAbet dictionary
    transcript = []
    for idx in predicted_indices[0]:  # Assuming batch_size=1
        if idx < len(ARPAbet):
            transcript.append(ARPAbet[idx.item()])
    
    return ''.join(transcript)

async def save_transcript_to_db(filename, transcript):
    """Save transcript and filename to MongoDB."""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["asr_db"]
    transcripts_collection = db["transcripts"]
    
    document = {"filename": filename, "transcript": transcript}
    await transcripts_collection.insert_one(document)