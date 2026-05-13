from fastapi import APIRouter, HTTPException, BackgroundTasks, Body, Query
import base64
import os
import uuid
from typing import List
from datetime import datetime
from models import ConversationSample, ConversationResponse
from database import db, get_next_conversation, USE_DATABASE

router = APIRouter()

# Directory for saving audio files
AUDIO_DIR = "uploads/audio/conversations"
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

@router.get("/random", response_model=ConversationSample)
async def get_random_conv(seenIds: List[str] = Query(default=[])):
    doc = await get_next_conversation(seen_ids=seenIds)
    if not doc:
        # Fallback if no conversations found or all seen
        return ConversationSample(
            id="mock_id",
            conversation="No new conversations available.",
            displayCount=0
        )
    
    return ConversationSample(
        id=doc.get("key_id") or str(doc.get("_id")) or str(uuid.uuid4()),
        conversation=doc.get("conversation", ""),
        displayCount=int(doc.get("display_count", 0) or 0)
    )

@router.post("/submit", status_code=201)
async def submit_response(response: ConversationResponse, background_tasks: BackgroundTasks):
    response_dict = response.dict()
    response_dict["created_at"] = datetime.utcnow().isoformat()
    
    # Handle audio data
    if response.responseType == "audio" and response.audioData:
        audio_path = save_audio(response.audioData, response.sampleId)
        response_dict["audio_path"] = audio_path
        # Remove large base64 string from DB document
        del response_dict["audioData"]
    
    if USE_DATABASE:
        await db.responses.insert_one(response_dict)
    else:
        print(f"Skipping DB insert for response: {response.sampleId}")
    
    return {"success": True, "message": "Response saved successfully"}
