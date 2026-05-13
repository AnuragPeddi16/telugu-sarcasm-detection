from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from database import db, USE_DATABASE

router = APIRouter()

class AdminLogin(BaseModel):
    username: str
    password: str

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

def get_demo_stats():
    return {
        "totalResponses": 247,
        "conversationResponses": 156,
        "validationResponses": 91,
        "sarcasticCount": 45,
        "nonSarcasticCount": 46,
        "demographics": {
            "regions": {
                "Hyderabad": 45,
                "Visakhapatnam": 32,
                "Guntur": 28,
                "Warangal": 24,
                "Nellore": 20,
                "Others": 98
            },
            "ageGroups": {
                "18-25": 89,
                "26-35": 76,
                "36-45": 45,
                "46-55": 24,
                "56-65": 10,
                "65+": 3
            },
            "education": {
                "Graduate": 102,
                "Post Graduate": 67,
                "Intermediate/12th": 45,
                "10th Pass": 23,
                "Doctorate": 8,
                "Below 10th": 2
            },
            "gender": {
                "Male": 132,
                "Female": 98,
                "Other": 12,
                "Prefer not to say": 5
            }
        }
    }

@router.post("/stats")
async def get_admin_stats(credentials: AdminLogin):
    if credentials.username != ADMIN_USERNAME or credentials.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not USE_DATABASE:
        return get_demo_stats()
    
    try:
        # Fetch all records
        responses = await db.responses.find().to_list(None)
        validations = await db.validations.find().to_list(None)
        
        conversation_count = len(responses)
        validation_count = len(validations)
        total_count = conversation_count + validation_count
        
        # Sarcastic and non-sarcastic counts only apply to validations
        sarcastic_count = sum(1 for v in validations if v.get("isSarcastic", v.get("is_sarcastic")) is True)
        non_sarcastic_count = validation_count - sarcastic_count
        
        demographics = {
            "regions": {},
            "ageGroups": {},
            "education": {},
            "gender": {}
        }
        
        all_records = responses + validations
        for record in all_records:
            demo = record.get("demographics", {})
            
            region = demo.get("region", "Unknown")
            demographics["regions"][region] = demographics["regions"].get(region, 0) + 1
            
            age = demo.get("ageGroup", "Unknown")
            demographics["ageGroups"][age] = demographics["ageGroups"].get(age, 0) + 1
            
            edu = demo.get("education", "Unknown")
            demographics["education"][edu] = demographics["education"].get(edu, 0) + 1
            
            gender = demo.get("gender", "Unknown")
            demographics["gender"][gender] = demographics["gender"].get(gender, 0) + 1

        # Calculate gold sample stats
        gold_total = 0
        gold_correct = 0
        for validation in validations:
            gold_stats = validation.get("sessionGoldStats", validation.get("session_gold_stats"))
            if gold_stats:
                gold_total += 1  # count validations where gold was processed
                # This could be more sophisticated if needed
                
        # we can just attach this extra stat
        result = {
            "totalResponses": total_count,
            "conversationResponses": conversation_count,
            "validationResponses": validation_count,
            "sarcasticCount": sarcastic_count,
            "nonSarcasticCount": non_sarcastic_count,
            "demographics": demographics
        }
        
        return result
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return get_demo_stats()

