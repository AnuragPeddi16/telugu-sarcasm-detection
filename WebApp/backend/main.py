from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from database import check_db_connection, seed_conversations_if_empty, seed_translations_if_empty, USE_DATABASE
from routes import conversations, validations, admin

app = FastAPI(title="Sarcasm Dataset Collection API")

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://telugu-sarcasm-website.vercel.app",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to check DB and seed data
@app.on_event("startup")
async def startup_db_client():
    if await check_db_connection():
        print("Connected to MongoDB")
        await seed_conversations_if_empty()
        await seed_translations_if_empty()
    else:
        if not USE_DATABASE:
             print("INFO: Database disabled. Running in mock/CSV mode.")
        else:
             print("WARNING: MongoDB connection failed. Application running in limited mode.")

# Include routers
app.include_router(conversations.router, prefix="/api/conversations", tags=["Conversations"])
app.include_router(validations.router, prefix="/api/validations", tags=["Validations"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

# Mount static files for audio if needed
app.mount("/static", StaticFiles(directory="uploads"), name="static")

@app.get("/")
def read_root():
    return {"message": "Welcome to Sarcasm Dataset Collection API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
