import os
import pandas as pd
import json
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict, Any
import random
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "sarcasm_data"
USE_DATABASE = os.getenv("USE_DATABASE", "True").lower() == "true"
USAGE_STATS_FILE = "data/usage_stats.json"

USE_GOLDS = os.getenv("USE_GOLDS", "True").lower() == "true"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

CONVERSATIONS_PATH = "data/llm_conversations_sarcasm_dataset.csv"
TRANSLATIONS_PATH = "data/translations_archita_te_utterence_only.csv"

def get_csv_conversations():
    try:
        df = pd.read_csv(CONVERSATIONS_PATH)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading conversations CSV: {e}")
        return []

def get_csv_translations():
    try:
        df = pd.read_csv(TRANSLATIONS_PATH)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading translations CSV: {e}")
        return []

def get_gold_translations():
    try:
        df = pd.read_csv("data/gold_translations.csv")
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading gold translations CSV: {e}")
        return []

def get_reported_ids():
    try:
        file_path = "data/bad_translations.csv"
        if not os.path.exists(file_path):
            return []
        df = pd.read_csv(file_path)
        return df['id'].astype(str).tolist()
    except Exception as e:
        print(f"Error loading bad translations CSV: {e}")
        return []

def load_usage_stats():
    if not os.path.exists(USAGE_STATS_FILE):
        return {}
    try:
        with open(USAGE_STATS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading usage stats: {e}")
        return {}

def save_usage_stats(stats):
    try:
        with open(USAGE_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Error saving usage stats: {e}")

async def increment_display_count(item_id: str, collection_name: str):
    if USE_DATABASE:
        await db[collection_name].update_one(
            {"key_id": item_id},
            {"$inc": {"display_count": 1}},
            upsert=True
        )
    else:
        stats = load_usage_stats()
        stats[item_id] = stats.get(item_id, 0) + 1
        save_usage_stats(stats)

async def get_usage_counts(collection_name: str, item_ids: List[str]) -> Dict[str, int]:
    if USE_DATABASE:
        # Fetch counts for relevant IDs
        cursor = db[collection_name].find({"key_id": {"$in": item_ids}})
        docs = await cursor.to_list(length=None)
        return {doc["key_id"]: doc.get("display_count", 0) for doc in docs}
    else:
        return load_usage_stats()

async def get_next_conversation(seen_ids: List[str] = []):
    """
    Selects a conversation based on 'Least Used' logic, excluding seen_ids.
    Always loads content from CSV, merging usage stats from DB or JSON.
    """
    # 1. Load all candidates from CSV
    items = get_csv_conversations()
    if not items:
        return None

    # 2. Prepare items with IDs
    for item in items:
        item["key_id"] = str(item.get("id"))

    # 3. Fetch usage stats
    all_ids = [item["key_id"] for item in items]
    usage_map = await get_usage_counts("conversations", all_ids)

    # 4. Merge stats
    for item in items:
        item["display_count"] = usage_map.get(item["key_id"], 0)

    # 5. Filter out seen items
    available_items = [item for item in items if item["key_id"] not in seen_ids]
    
    if not available_items:
        # If all seen, fallback to all items (recycle)
        available_items = items
    
    # 6. Sort by display_count ascending
    available_items.sort(key=lambda x: x.get("display_count", 0))

    # 7. Randomize among top N least used
    top_n = max(1, min(10, len(available_items)))
    candidates = available_items[:top_n]
    
    selected = random.choice(candidates)
    
    item_id = selected["key_id"]
    await increment_display_count(item_id, "conversations")
    
    return {
        "key_id": item_id,
        "_id": str(selected.get("_id") or item_id),
        "conversation": selected.get("conversation", ""),
        "display_count": selected.get("display_count", 0) + 1
    }

async def get_next_translation(seen_ids: List[str] = [], is_gold: bool = False):
    """
    Selects a translation based on 'Least Used' logic, or returns a gold sample.
    Always loads content from CSV, merging usage stats from DB or JSON.
    """
    if is_gold and USE_GOLDS:
        gold_items = get_gold_translations()
        if gold_items:
            selected = random.choice(gold_items)
            return {
                "key_id": str(selected.get("id") or str(uuid.uuid4())),
                "_id": str(selected.get("id") or "gold_id"),
                "utterance": selected.get("utterance", selected.get("utterence", "")),
                "is_gold": True,
                "is_sarcastic": selected.get("is_sarcastic", False),
                "display_count": 0
            }

    # 1. Load all candidates from CSV
    items = get_csv_translations()
    if not items:
        return None

    # 2. Prepare items with IDs
    for item in items:
        item["key_id"] = str(item.get("id"))

    # 3. Fetch usage stats
    all_ids = [item["key_id"] for item in items]
    usage_map = await get_usage_counts("translations", all_ids)

    # 4. Merge stats
    for item in items:
        item["display_count"] = usage_map.get(item["key_id"], 0)

    # 5. Filter out seen items and reported bad items
    reported_ids = get_reported_ids()
    excluded_ids = set(seen_ids) | set(reported_ids)
    
    available_items = [item for item in items if item["key_id"] not in excluded_ids]
    
    if not available_items:
        # Fallback if everything is seen/excluded (though reported shouldn't be recycled ideally)
        # We can recycle seen items but filter reported ones again if strictly needed
        available_items = [item for item in items if item["key_id"] not in reported_ids]
        if not available_items:
             # If strictly everything is reported, well, we return nothing or raw
             available_items = items

    # 6. Sort by display_count ascending
    available_items.sort(key=lambda x: x.get("display_count", 0))

    # 7. Randomize among top N least used
    top_n = max(1, min(10, len(available_items)))
    candidates = available_items[:top_n]
    
    selected = random.choice(candidates)
    
    item_id = selected["key_id"]
    await increment_display_count(item_id, "translations")

    return {
        "key_id": item_id,
        "_id": str(selected.get("_id") or item_id),
        "utterance": selected.get("utterance", selected.get("utterence", "")),
        "display_count": selected.get("display_count", 0) + 1
    }

async def report_bad_translation(item_id: str, reason: str):
    # Log to bad_translations.csv
    file_path = "data/bad_translations.csv"
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
            
        file_exists = os.path.isfile(file_path)
        with open(file_path, "a", encoding='utf-8') as f:
            if not file_exists:
                f.write("id,reason,date\n")
            f.write(f"{item_id},{reason},{datetime.now().isoformat()}\n")
    except Exception as e:
        print(f"Error logging bad translation: {e}")

async def check_db_connection():
    if not USE_DATABASE:
        return False
    try:
        # Check connection status
        await client.admin.command('ping')
        return True
    except Exception as e:
        print(f"MongoDB connection check failed: {e}")
        return False

async def seed_conversations_if_empty():
    """
    No-op: Conversations are read directly from CSV files.
    """
    pass

async def seed_translations_if_empty():
    """
    No-op: Translations are read directly from CSV files.
    """
    pass

