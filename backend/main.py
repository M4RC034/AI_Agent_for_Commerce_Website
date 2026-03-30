import os
import uvicorn
import io
import json
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

try:
    from groq import Groq
except ImportError:
    Groq = None

# We use python-dotenv indirectly or normally just reading OS env
from .search_engine import CatalogSearchEngine

app = FastAPI(title="AI Commerce Agent Backend")

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold singletons
search_engine = None
groq_client = None

@app.on_event("startup")
def startup_event():
    global search_engine, groq_client
    
    # Load paths based on expected directory structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(BASE_DIR, "data", "processed", "catalog.index")
    catalog_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_catalog_with_images.jsonl")
    
    if os.path.exists(index_path) and os.path.exists(catalog_path):
        search_engine = CatalogSearchEngine(index_path, catalog_path)
    else:
        print("WARNING: FAISS Index or Catalog JSONL missing! Please run Day 1 scripts.")
        
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Check if user put it in api_key.txt
    api_key_path = os.path.join(BASE_DIR, "api_key.txt")
    if not groq_api_key and os.path.exists(api_key_path):
        with open(api_key_path, "r") as f:
            groq_api_key = f.read().strip(" .\n\r")
            
    if groq_api_key and Groq:
        groq_client = Groq(api_key=groq_api_key)
        print("Groq Client API Key Loaded Successfully!")
    else:
        print("WARNING: GROQ_API_KEY not found. Please set the environment variable or place it in api_key.txt")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Aura Backend is running!"}

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    The Single 'Vision-via-Retrieval' Endpoint.
    1. Visual Retrieval (or Text Retrieval).
    2. DeepSeek RAG reasoning over the metadata.
    3. Return structured API payload.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API Key not configured.")
        
    try:
        context_items = []
        if image:
            # Step 1a: Use the Eyes (OpenCLIP Visual Search)
            image_bytes = await image.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            context_items = search_engine.search_by_image(pil_img, k=3)
        else:
            # Step 1b: If no image, fallback to vector text search based on the message
            context_items = search_engine.search_by_text(message, k=5)
            
        # Format the FAISS results to feed to the "Brain"
        context_str = "VISUAL/METADATA RAG SEARCH RESULTS:\n\n"
        for i, item in enumerate(context_items):
            context_str += f"[Result {i+1}] Title: {item.get('title')}\n"
            context_str += f"Category: {item.get('category')} | Price: ${item.get('price')}\n\n"
            
        # Step 2: Use the Brain (DeepSeek-R1-Distill-Llama-70B on Groq)
        system_prompt = (
            "You are Aura, an elite AI shopping assistant for a commerce website. "
            "You must rely ONLY on the VISUAL/METADATA RAG SEARCH RESULTS provided in the prompt to make recommendations. "
            "Do NOT invent products. The user uploaded an image or query, and the backend engine visually found the products below. "
            "Explain to the user why these specific items match their request, or help them decide based on constraints."
        )
        
        user_prompt = f"User Message: {message}\n\n{context_str}"
        
        # Deepseek R1 API call
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.6,
        )
        
        agent_raw_reply = chat_completion.choices[0].message.content
        
        # Format output: Deepseek uses <think> tags. Let's strip them from the frontend view 
        # or leave them for debugging, but typically for a retail bot we clean the <think> out.
        formatted_reply = agent_raw_reply
        if "</think>" in formatted_reply:
            formatted_reply = formatted_reply.split("</think>")[-1].strip()
        
        return {
            "agent_response": formatted_reply,
            "thoughts": agent_raw_reply.split("</think>")[0].replace("<think>", "").strip() if "</think>" in agent_raw_reply else "",
            "products": context_items
        }

    except Exception as e:
        print(f"Error during chat handling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
