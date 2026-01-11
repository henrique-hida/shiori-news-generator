import os
import json
import asyncio
import certifi
import logging
import urllib.parse
import datetime
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from ddgs import DDGS

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = 'gemini-2.5-flash' 
    REQUEST_INTERVAL_SECONDS = 10
    
    SUBJECTS = [
        "Politics", "Economy & Business", "Technology", 
        "Health & Science", "Sports", "Entertainment & Culture"
    ]
    
    LANGUAGES = {
        "English": "en",
        "Portuguese": "pt",
        "Spanish": "es"
    }

class RateLimiter:
    def __init__(self, interval_seconds: float):
        self.interval = interval_seconds
        self.last_call_time = 0
        self.lock = asyncio.Lock()

    async def wait_for_slot(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            wait_time = self.interval - elapsed
            
            if wait_time > 0:
                logging.info(f"⏳ Rate Limit: Sleeping for {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            
            self.last_call_time = time.time()

class FirebaseService:
    def __init__(self, key_path: str):
        if not firebase_admin._apps:
            firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS")
            if firebase_creds_json:
                cred = credentials.Certificate(json.loads(firebase_creds_json))
            else:
                cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def save_subject_news(self, date_str: str, subject_slug: str, data: Dict[str, Any]):
        try:
            doc_ref = self.db.collection("news").document(date_str).collection("subjects").document(subject_slug)
            doc_ref.set(data, merge=True)
            logging.info(f"✅ Saved Firestore: {subject_slug}")
        except Exception as e:
            logging.error(f"❌ Firestore Save Error: {e}")

class ImageService:
    def get_thumbnail_url(self, subject: str) -> str:
        try:
            time.sleep(1) 
            with DDGS() as ddgs:
                results = list(ddgs.images(subject, max_results=1, safesearch='on'))
            if results:
                return results[0]['image']
            return self._get_placeholder(subject)
        except Exception:
            return self._get_placeholder(subject)

    def _get_placeholder(self, text: str) -> str:
        sanitized = urllib.parse.quote(text)
        return f"https://placehold.co/600x400/1e293b/ffffff/png?text={sanitized}"

class NewsGenerator:
    def __init__(self, api_key: str, model_name: str, rate_limiter: RateLimiter):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name) 
        self.rate_limiter = rate_limiter

    def _get_news_context(self, subject: str, date_str: str) -> dict:
        # trusted_sites = "site:cnn.com OR site:bbc.com OR site:reuters.com OR site:bloomberg.com" OR "site:g1.globo.com"
        # search_query = f"{subject} news {date_str} ({trusted_sites})"

        search_query = f"{subject} news {date_str}"
        context_text = ""
        sources = []
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, region='wt-wt', safesearch='on', max_results=7))
            
            for i, res in enumerate(results):
                context_text += f"SOURCE {i+1}: {res['title']}\nSUMMARY: {res['body']}\n\n"
                sources.append(res['href'])
                
        except Exception as e:
            logging.error(f"⚠️ Search Error for {subject}: {e}")
            
        return {"text": context_text, "links": sources}

    def _create_prompt(self, subject: str, language: str, date_str: str, context_text: str) -> str:
        return f"""
        Act as a news aggregator. 
        Here is the raw news data gathered TODAY ({date_str}) for '{subject}':
        
        === BEGIN RAW DATA ===
        {context_text}
        === END RAW DATA ===

        INSTRUCTIONS:
        1. Summarize these specific events. 
        2. **NO GENERIC FILLER**: Do not write "Markets are digesting..." or "Political landscape is shifting...". Use the specific facts from the Raw Data above.
        3. **BANNED INTROS**: NEVER start with "Today,", "On {date_str},". Start directly with the event.
        4. **Language**: Write in {language}.
        5. **Schema**: Return ONLY JSON.
        
        JSON Schema:
        {{
            "styles": {{
                "impartial": {{ 
                    "title": "Neutral Headline",
                    "durations": {{ "fast": "50 words max", "standard": "100 words", "deep": "250 words" }} 
                }},
                "informal": {{ "title": "Casual Headline", "durations": {{ "fast": "...", "standard": "...", "deep": "..." }} }},
                "analytic": {{ "title": "Data-focused Headline", "durations": {{ "fast": "...", "standard": "...", "deep": "..." }} }},
                "funny": {{ "title": "Witty Headline", "durations": {{ "fast": "...", "standard": "...", "deep": "..." }} }}
            }}
        }}
        """

    async def generate_content_async(self, subject: str, language: str, date_str: str) -> Optional[Dict]:
        await self.rate_limiter.wait_for_slot()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_content_sync, subject, language, date_str)

    def _generate_content_sync(self, subject: str, language: str, date_str: str) -> Optional[Dict]:
        news_data = self._get_news_context(subject, date_str)
        
        if not news_data["text"]:
            logging.warning(f"⚠️ No news found for {subject}")
            return None

        prompt = self._create_prompt(subject, language, date_str, news_data["text"])
        
        try:
            response = self.model.generate_content(prompt)
            text_response = response.text
            clean_json = text_response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            data['sources'] = news_data["links"]

            return data
        except Exception as e:
            logging.error(f"❌ API Error for {subject} ({language}): {e}")
            return None

# --- MAIN CONTROLLER ---

async def process_subject(subject: str, date_str: str, generator: NewsGenerator, image_service: ImageService, firebase: FirebaseService):
    logging.info(f"🔹 Starting Subject: {subject}")

    lang_keys = []
    tasks = []
    for lang_name, lang_code in Config.LANGUAGES.items():
        lang_keys.append(lang_code)
        tasks.append(generator.generate_content_async(subject, lang_name, date_str))
    
    results = await asyncio.gather(*tasks)
    image_search_query = subject 
    
    for res in results:
        if res and 'styles' in res and 'impartial' in res['styles']:
            headline = res['styles']['impartial']['title']
            if headline:
                image_search_query = headline
                break
    
    logging.info(f"📷 Searching Image for: '{image_search_query}'")
    loop = asyncio.get_running_loop()
    thumb_url = await loop.run_in_executor(None, image_service.get_thumbnail_url, image_search_query)
    
    subject_payload = {
        "category": subject,
        "thumbUrl": thumb_url,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "content": {}
    }
    
    for i, result in enumerate(results):
        if result:
            subject_payload["content"][lang_keys[i]] = result
            
    doc_id = subject.lower().replace(" & ", "-").replace(" ", "-")
    firebase.save_subject_news(date_str, doc_id, subject_payload)

async def main():
    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    firebase = FirebaseService(Config.FIREBASE_KEY_PATH)
    image_service = ImageService()
    rate_limiter = RateLimiter(Config.REQUEST_INTERVAL_SECONDS)
    generator = NewsGenerator(Config.GEMINI_API_KEY, Config.MODEL_NAME, rate_limiter)
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    logging.info(f"--- Starting News Generation for {today} ---")
    logging.info(f"⚠️  Rate Limit Active: 1 request every {Config.REQUEST_INTERVAL_SECONDS} seconds.")
    logging.info(f"📊 Total Requests to be made: {len(Config.SUBJECTS) * len(Config.LANGUAGES)}")

    subject_tasks = [
        process_subject(subj, today, generator, image_service, firebase) 
        for subj in Config.SUBJECTS
    ]
    
    await asyncio.gather(*subject_tasks)
    logging.info("--- All Done ---")

if __name__ == "__main__":
    asyncio.run(main())