import torch
import os
import re
import uvicorn
import io
import json
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Automatically load API keys from .env or api_key.env
import base64
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))
load_dotenv(os.path.join(BASE_DIR, 'backend', 'api_key.env'))

try:
    from groq import Groq
except ImportError:
    Groq = None

# We use python-dotenv indirectly or normally just reading OS env
from search_engine import CatalogSearchEngine

app = FastAPI(
    title="Al — AI Commerce Agent API",
    description=(
        "A multimodal AI shopping assistant that supports **text-based product recommendations**, "
        "**image-based product search**, and **general conversation**.\n\n"
        "- **Text only →** MiniLM encoder + FAISS text index\n"
        "- **Image only →** OpenCLIP encoder + FAISS visual index + Zero-Shot Distractor Gate\n"
        "- **Text + Image →** Late Fusion via Reciprocal Rank Fusion (RRF)\n"
        "- **General chat →** Direct LLM conversation (no retrieval)\n\n"
        "Reasoning powered by Llama-3.1-8b-instant on Groq."
    ),
    version="1.0.0",
)

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
        
    raw_key = os.environ.get("GROQ_API_KEY", "")
    groq_api_key = raw_key.strip().strip('"').strip("'")
    if groq_api_key and Groq:
        groq_client = Groq(api_key=groq_api_key)
    else:
        print("WARNING: GROQ_API_KEY environment variable not set. LLM inference will fail.")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Al Backend is running!"}

@app.get("/api/statistics")
def get_statistics():
    """Return category distribution for the pie chart."""
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
        
    counts = search_engine.df['product_category'].value_counts().to_dict()
        
    return {"categories": counts}

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    history_json: str = Form("[]")
):
    """
    The Single 'Vision-via-Retrieval' Endpoint with multi-turn support.
    Accepts an optional history_json field containing previous conversation turns.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API Key not configured.")
    
    # Parse conversation history (sent as JSON string via FormData)
    try:
        history = json.loads(history_json)
        # Keep only the last 5 turns to stay within token limits
        history = history[-5:]
    except (json.JSONDecodeError, TypeError):
        history = []
        
    try:
        # ── Intent Detection: General Conversation vs Product Query ──
        # If no image is attached, check whether the message is a general
        # conversational question (no product intent) and handle it directly
        # without hitting the retrieval pipeline.
        if not image and _is_general_conversation(message):
            return await _handle_general_conversation(message, history)
        
        context_items = []
        is_default_msg = message == "Please find items visually similar to this image."
        
        # --- Metadata Pre-Filtering ---
        # Extract dynamic constraints (like max price) before searching
        max_price = None
        if message and not is_default_msg:
            # Look for phrasing like "under $100" or "less than 50"
            price_match = re.search(r'(?:under|below|less than|cheaper than)\s*\$?\s*(\d+(?:\.\d{1,2})?)', message, re.IGNORECASE)
            if price_match:
                max_price = float(price_match.group(1))
        
        if image and message and not is_default_msg:
            # Both provided -> Late Fusion Hybrid Search
            image_bytes = await image.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            context_items = search_engine.hybrid_search(message, pil_img, k=5, max_price=max_price)
            
        elif image:
            # Only Image provided -> OpenCLIP
            image_bytes = await image.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            context_items = search_engine.search_by_image(pil_img, k=5, max_price=max_price)
            
        else:
            # Only Text provided -> MiniLM
            context_items = search_engine.search_by_text(message, k=5, max_price=max_price)
            
        # Format the FAISS results to feed to the "Brain"
        valid_cats = [
            'Laptops', 'Phones', 'Headphones', 'Chargers & Cables', 'Cameras', 
            'Storage', 'Smart Home', 'TV & Display', 'Power & Batteries', 
            'Networking', 'Wearables', 'Speakers', 'Printers & Scanners', 'Gaming',
            'Computers & Accessories', 'Computer Accessories & Peripherals', 'Monitors'
        ]

        context_str = "VISUAL/METADATA RAG SEARCH RESULTS:\n\n"
        final_products = []
        for i, item in enumerate(context_items):
            # Strict safety net: ONLY pass electronic categories to the LLM
            if item.get('product_id') != 'NONE' and item.get('category', '') in valid_cats:
                final_products.append(item)
                res_index = len(final_products)
                title = item.get('title', '')
                context_str += f"[Result {res_index}] ID: {item.get('product_id')} | Title: {title}\n"
                context_str += f"Category: {item.get('category')} | Price: ${item.get('price')}\n"
                
                # --- Attribute Extraction Hint ---
                storage_match = re.search(r'(\d+)\s*(?:TB|GB)', title, re.IGNORECASE)
                if storage_match:
                    context_str += f"HINT: Result {res_index} has {storage_match.group(0).upper()} Storage.\n"
                    
                context_str += "\n"
                
        # Handle the Empty Context Edge Case (OOD Rejection)
        # --- Handle the Empty Context Edge Case (OOD Rejection) ---
        if len(final_products) == 0:
            if image:
                image_description = "this item"
                try:
                    image.file.seek(0)
                    image_bytes = await image.read()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    vision_completion = groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Identify the primary object in this image. Answer with only the name of the object (e.g., 'a dishwasher' or 'a wooden table')."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ],
                        model="llama-3.2-11b-vision-preview",
                    )
                    image_description = vision_completion.choices[0].message.content.strip().lower()
                    
                except Exception as vision_err:
                    print(f"Vision analysis failed: {str(vision_err)}")
                
                # Image OOD Response
                return {
                    "agent_response": (
                        f"According to the image you provided, you are likely looking for {image_description}. "
                        "I'm sorry, but this appears to be outside of our catalog. We specialize exclusively "
                        "in electronics, and I could not confidently find any safe, relevant matches."
                    ),
                    "products": []
                }
            else:
                # Text-Only OOD Response
                return {
                    "agent_response": (
                        "I'm sorry, but it looks like you are searching for items outside of our catalog. "
                        "We specialize exclusively in electronics, and I couldn't find any relevant matches "
                        "for your request. Could I help you find a laptop, phone, or accessory instead?"
                    ),
                    "products": []
                }
            
        # Step 2: Use the Brain
        system_prompt = (
            "You are Al, an elite AI shopping assistant for a commerce website.\n"
            "When a user asks to compare products or find the 'best/most' of an attribute (like storage, price, or speed):\n"
            "1. Look at the VISUAL/METADATA RAG SEARCH RESULTS provided.\n"
            "2. Explicitly compare the values (e.g., 'The Seagate Drive has 12TB while the Western Digital has 8TB').\n"
            "3. Even if the items are 'Internal' rather than 'External,' answer the user's specific question about capacity first, "
            "then politely mention the type mismatch (e.g., 'Note: These are internal drives, not external').\n\n"
            "CRITICAL RULES:\n"
            "1. MUST START YOUR RESPONSE EXACTLY WITH: \"According to the information you provide, these are the products that you may be interested:\"\n"
            "2. Then ONLY list the relevant products that answer the user's specific query. Do NOT list, mention, or explain any products you ignored or found irrelevant.\n"
            "3. Explain why the relevant items match their request succinctly.\n"
            "4. VERY IMPORTANT: You must include the exact ID of the product (e.g., prod_10) in parentheses next to its name when you describe it.\n"
            "5. TRUST THE CATEGORY: If the metadata says a product is in the 'Headphones' category, do not re-categorize it based on other keywords in the description (like 'battery' or 'cable')."
        )
        
        user_prompt = f"User Message: {message}\n\n{context_str}"
        
        # Build message array with conversation history for multi-turn context
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            role = turn.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_prompt})
        
        # Groq Llama 3 API call
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.6,
        )
        
        agent_raw_reply = chat_completion.choices[0].message.content

        # --- ROBUST ID EXTRACTION ---
        # Instead of relying on strict formatting, we cross-reference the text 
        # to see which product IDs the LLM actually mentioned anywhere in its response.
        selected_ids = set()
        agent_reply_lower = agent_raw_reply.lower()
        
        for p in final_products:
            pid = str(p.get('product_id', '')).lower()
            if pid and pid in agent_reply_lower:
                selected_ids.add(pid)
                
        # Filter final_products based on the IDs the LLM actually mentioned
        if selected_ids:
            final_products = [
                p for p in final_products 
                if str(p.get('product_id', '')).lower() in selected_ids
            ]
        else:
            # If the LLM completely failed to mention ANY IDs, 
            # we return an empty list so the UI doesn't show hallucinated items.
            final_products = []
            
        # Clean the response for the UI (Remove the IDs from the text if they were added at the end)
        clean_reply = re.sub(r'IDs?:\s*\[.*?\]', '', agent_raw_reply, flags=re.IGNORECASE).strip()
        
        if "</think>" in clean_reply:
            clean_reply = clean_reply.split("</think>")[-1].strip()
        
        return {
            "agent_response": clean_reply,
            "products": final_products
        }

    except Exception as e:
        print(f"Error during chat handling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Intent Detection Helpers ──────────────────────────────────────────

# Conversational patterns that indicate the user is NOT looking for products.
_GENERAL_PATTERNS = [
    r"\b(what(?:'s| is) your name)\b",
    r"\b(who are you)\b",
    r"\b(what can you do)\b",
    r"\b(how are you)\b",
    r"\b(tell me about yourself)\b",
    r"\b(what are you)\b",
    r"\b(help me|what do you offer)\b",
    r"\b(hello|hi|hey|good morning|good afternoon|good evening|good night)\b",
    r"\b(thank(?:s| you))\b",
    r"\b(bye|goodbye|see you)\b",
    r"\b(difference between you)\b",
    r"\b(other ai models?)\b",
    r"\b(how do you work)\b",
    r"\b(who made you|who created you)\b",
    r"\b(are you an ai|are you a bot)\b",
    r"\b(what model)\b"
]
_GENERAL_RE = re.compile("|".join(_GENERAL_PATTERNS), re.IGNORECASE)

# Product-intent signals — if any of these appear, treat it as a product query
# even if it also matches a greeting (e.g. "hi, recommend me headphones").
_PRODUCT_SIGNALS = [
    r"\b(recommend|suggest|find|search|show|looking for|need|want|buy|compare)\b",
    r"\b(cheap|budget|best|top|affordable|premium|under \$?\d+)\b",
    r"\b(laptop|phone|headphone|earbuds|charger|cable|camera|speaker|monitor|tv|watch|printer|keyboard|mouse|tablet)s?\b",
]
_PRODUCT_RE = re.compile("|".join(_PRODUCT_SIGNALS), re.IGNORECASE)


def _is_general_conversation(message: str) -> bool:
    """Return True if the message is a general/conversational query with
    no product-search intent."""
    text = message.strip()
    # Very short messages with no product keywords are likely greetings
    if len(text.split()) <= 4 and _GENERAL_RE.search(text) and not _PRODUCT_RE.search(text):
        return True
    # Longer messages: only treat as general if they match a pattern AND
    # have zero product signals
    if _GENERAL_RE.search(text) and not _PRODUCT_RE.search(text):
        return True
    return False


async def _handle_general_conversation(message: str, history: list = None) -> dict:
    """Send the message directly to the LLM with a conversational persona,
    bypassing the retrieval pipeline entirely. Supports multi-turn context."""
    system_prompt = (
        "You are Al, a friendly and helpful AI shopping assistant for an electronics "
        "commerce website. You specialize in electronics such as laptops, phones, "
        "headphones, cameras, speakers, TVs, and more.\n\n"
        "When users greet you or ask general questions, respond warmly and concisely. "
        "Let them know you can help them find electronics by describing what they need "
        "in text, or by uploading a photo of a product they want to find.\n\n"
        "CRITICAL INSTRUCTION: If the user asks 'what do you sell', 'what do you have', or similar inventory questions, "
        "you MUST explicitly answer: 'We mainly focus on electronics. Please check the \"About\" section to see what kinds of items we have.'\n\n"
        "If asked about your identity, underlying AI model, or how you differ from other AIs, explicitly explain that you are 'Al', "
        "a specialized multimodal RAG shopping interface designed natively for this commerce catalog. You blend visual vector search "
        "and semantic text retrieval, rather than just being a general conversational chatbot.\n\n"
        "Keep responses short (2-4 sentences). Be personable but professional."
    )

    # Build message array with conversation history
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history:
            role = turn.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})

    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant",
        temperature=0.7,
    )

    reply = chat_completion.choices[0].message.content.strip()
    # Clean any stray think tags
    if "</think>" in reply:
        reply = reply.split("</think>")[-1].strip()

    return {"agent_response": reply, "products": []}


from fastapi.staticfiles import StaticFiles

# This mounts your frontend folder to the root URL. 
# html=True tells it to automatically load index.html when you visit /
frontend_path = os.path.join(BASE_DIR, "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
