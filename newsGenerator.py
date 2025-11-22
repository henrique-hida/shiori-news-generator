import os
import json
import time
import datetime
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
class Config:
    FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = 'gemini-2.5-flash'
    
    SUBJECTS = [
        "Politics", "Economy & Business", "Technology", 
        "Health & Science", "Sports", "Entertainment & Culture"
    ]
    
    LANGUAGES = {
        "English": "eng",
        "Portuguese": "por",
        "Spanish": "spa"
    }

# --- SERVICES ---
class FirebaseService:
    def __init__(self, key_path: str):
        if not firebase_admin._apps:
            firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS")
            
            if firebase_creds_json:
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
            else:
                cred = credentials.Certificate(key_path)
                
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def save_subject_news(self, date_str: str, subject_slug: str, data: Dict[str, Any]):
        try:
            doc_ref = self.db.collection("news").document(date_str).collection("subjects").document(subject_slug)
            doc_ref.set(data)
            logging.info(f"✅ Saved {subject_slug} to Firestore.")
        except Exception as e:
            logging.error(f"Failed to save to Firestore: {e}")

class NewsGenerator:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    def _create_prompt(self, subject: str, language: str, date_str: str) -> str:
        return f"""
        You are a professional news aggregator. 
        Identify the most important news stories for today ({date_str}) regarding the subject: '{subject}'.
        
        Output a JSON object with this exact schema:
        {{
            "impartial": {{ "fast": "...", "standard": "...", "deep": "..." }},
            "informal": {{ "fast": "...", "standard": "...", "deep": "..." }},
            "analytic": {{ "fast": "...", "standard": "...", "deep": "..." }},
            "funny": {{ "fast": "...", "standard": "...", "deep": "..." }}
        }}

        Directives:
        1. Language: Write in {language}.
        2. Content: Combine top stories into one narrative.
        3. Durations: Fast (1 min), Standard (5 min), Deep (10 min).
        """

    def generate_content(self, subject: str, language: str, date_str: str) -> Optional[Dict]:
        prompt = self._create_prompt(subject, language, date_str)
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logging.error(f"❌ API Error for {subject} ({language}): {e}")
            return None

# --- MAIN CONTROLLER ---
def main():
    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    firebase = FirebaseService(Config.FIREBASE_KEY_PATH)
    generator = NewsGenerator(Config.GEMINI_API_KEY, Config.MODEL_NAME)
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    logging.info(f"--- Starting News Generation for {today} ---")

    for subject in Config.SUBJECTS:
        logging.info(f"Processing Subject: {subject}...")
        
        subject_payload = {
            "category": subject,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "content": {}
        }
        
        for lang_name, lang_code in Config.LANGUAGES.items():
            logging.info(f"  - Generating {lang_name}...")
            
            content = generator.generate_content(subject, lang_name, today)
            
            if content:
                subject_payload["content"][lang_code] = content
            
            time.sleep(2)
        
        doc_id = subject.lower().replace(" & ", "-").replace(" ", "-")
        firebase.save_subject_news(today, doc_id, subject_payload)

if __name__ == "__main__":
    main()