from fastapi import APIRouter, HTTPException, BackgroundTasks, Body, Query
import base64
import os
import uuid
from typing import List
from datetime import datetime
from models import TranslationSample, ValidationResponse, ReportRequest
from database import db, get_next_translation, report_bad_translation, USE_DATABASE

router = APIRouter()

# Directory for saving audio files
AUDIO_DIR = "uploads/audio/validations"
os.makedirs(AUDIO_DIR, exist_ok=True)

def save_audio(base64_string: str, sample_id: str) -> str:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        file_path = os.path.join(AUDIO_DIR, f"{sample_id}_{uuid.uuid4()}.wav")
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(base64_string))
        return file_path
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None

@router.get("/random", response_model=TranslationSample)
async def get_random_trans(seenIds: List[str] = Query(default=[]), isGold: bool = False):
    doc = await get_next_translation(seen_ids=seenIds, is_gold=isGold)
    
    if not doc:
        return TranslationSample(
            id="mock_trans_id",
            utterance="No new translations available. Please check back later.",
            displayCount=0,
            isGold=False
        )

    return TranslationSample(
        id=doc.get("key_id") or str(doc.get("_id")) or str(uuid.uuid4()),
        utterance=doc.get("utterance", ""),
    )

@router.post("/report", status_code=200)
async def report_translation(report: ReportRequest):
    await report_bad_translation(report.id, report.reason)
    return {"status": "reported", "id": report.id}

@router.post("/submit", status_code=201)
async def submit_validation(response: ValidationResponse, background_tasks: BackgroundTasks):
    validation_dict = response.dict()
    validation_dict["created_at"] = datetime.utcnow().isoformat()
    
    # Handle audio data if sarcastic example is audio
    if response.responseType == "audio" and response.audioData:
        audio_path = save_audio(response.audioData, response.sampleId)
        validation_dict["audio_path"] = audio_path
        # Remove large base64 string from DB document
        del validation_dict["audioData"]
    
    if USE_DATABASE:
        await db.validations.insert_one(validation_dict)
    else:
        print(f"Skipping DB insert for validation: {response.sampleId}")
    
    return {"success": True, "message": "Validation saved successfully"}
