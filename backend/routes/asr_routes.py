from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.services.asr_service import process_audio
from motor.motor_asyncio import AsyncIOMotorClient

asr_router = APIRouter()

@asr_router.post("/asr/")
async def asr_endpoint(audio_file: UploadFile = File(...)):
    try:
        transcript = await process_audio(audio_file)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
