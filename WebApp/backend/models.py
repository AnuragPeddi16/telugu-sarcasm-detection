from typing import Optional, List
from pydantic import BaseModel, Field, BeforeValidator
from typing_extensions import Annotated

# Helper for MongoDB ObjectId
PyObjectId = Annotated[str, BeforeValidator(str)]

class DemographicInfo(BaseModel):
    region: str
    gender: str
    education: str
    ageGroup: str

class ConversationSample(BaseModel):
    id: str
    conversation: str
    displayCount: int = 0

class TranslationSample(BaseModel):
    id: str
    utterance: str
    displayCount: int = 0
    isGold: bool = False
    correctIsSarcastic: Optional[bool] = None

class ReportRequest(BaseModel):
    id: str
    reason: str
    details: Optional[str] = None


class ConversationResponse(BaseModel):
    sampleId: str
    originalConversation: str
    sarcasticResponse: str
    nonSarcasticResponse: str
    demographics: DemographicInfo
    responseType: str = Field(pattern="^(text|audio)$")
    audioData: Optional[str] = None  # Base64 string if responseType is audio

class SessionGoldStats(BaseModel):
    total: int
    correct: int

class ValidationResponse(BaseModel):
    sampleId: str
    originalUtterance: str
    isSarcastic: bool
    sarcasticExample: Optional[str] = None
    demographics: DemographicInfo
    responseType: Optional[str] = Field(None, pattern="^(text|audio)$")
    audioData: Optional[str] = None
    sessionGoldStats: Optional[SessionGoldStats] = None

# Database Models

class ConversationDB(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    key_id: str  # Original ID from CSV or unique identifier
    conversation: str
    display_count: int = 0
    language: str = "te" # Default to Telugu
    source: str = "csv" # or "generated"

class TranslationDB(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    key_id: str
    utterance: str
    display_count: int = 0
    language: str = "te"

class UserResponseDB(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    sample_id: str
    original_conversation: str
    sarcastic_response: str
    non_sarcastic_response: str
    demographics: DemographicInfo
    response_type: str
    audio_path: Optional[str] = None # Path to stored audio file
    created_at: str # ISO format date

class UserValidationDB(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    sample_id: str
    original_utterance: str
    is_sarcastic: bool
    sarcastic_example: Optional[str] = None
    demographics: DemographicInfo
    response_type: Optional[str] = None
    audio_path: Optional[str] = None
    session_gold_stats: Optional[SessionGoldStats] = None
    created_at: str
