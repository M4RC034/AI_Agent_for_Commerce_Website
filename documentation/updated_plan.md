# Architecture Update: Dual-Index Late-Fusion with Hybrid AI

## 1. Overview of the Pivot

In our initial plan (`initial_plan.md`), the architecture outlined using a single Multi-modal LLM (e.g., GPT-4o or Claude 3.5 Sonnet) as the primary engine to process both text constraints and direct image uploads.

After further review, we pivoted to a **"Vision-via-Retrieval"** RAG (Retrieval-Augmented Generation) pipeline. The core difference is that the final Reasoning Agent (LLM) is entirely **blind** to the image itself. Instead, a specialized Machine Learning pipeline evaluates the image locally and injects explicit product matches as text context into the LLM prompt.

This document reflects the **current state** of the system after multiple iterations of optimization, including:
- A **Dual-Index Late-Fusion** architecture for intelligent query routing.
- An **Intent-Product Alignment Layer** that synchronizes LLM reasoning with frontend rendering.
- An **Out-of-Distribution (OOD) Classifier** with a Cloud Vision Fallback for graceful rejection.
- A **Statistics Dashboard** for catalog analytics.

---

## 2. Current Architecture

### The Dual-Index "Late Fusion" Pipeline

The system now maintains **two separate FAISS vector indices**, each powered by a different encoder, and intelligently routes queries based on input type:

| Input Type | Encoder | Index | Latency |
| :--- | :--- | :--- | :--- |
| **Text Only** | `all-MiniLM-L6-v2` (384D) | `text_catalog.index` | ~5ms |
| **Image Only** | `OpenCLIP ViT-B-16-SigLIP` (768D) | `catalog.index` | ~50ms |
| **Text + Image** | Both encoders → RRF fusion | Both indices | ~60ms |

### The Request Workflow

1. **The Request:** User enters a text query, uploads an image, or both via the `/api/chat` endpoint.
2. **Smart Routing (`main.py`):** The backend detects which inputs are present and dispatches to the correct search method:
   - **Text only →** `search_by_text()` using the lightweight MiniLM encoder.
   - **Image only →** `search_by_image()` using OpenCLIP with a Zero-Shot Distractor Gate.
   - **Both →** `hybrid_search()` using Reciprocal Rank Fusion (RRF) to merge results from both indices.
3. **Category Safety Net (`main.py`):** Results are filtered through a `valid_cats` whitelist to prevent non-electronic products from reaching the LLM.
4. **OOD Rejection with Cloud Vision Fallback:** If zero valid products survive the filter, the system calls **Llama 3.2 Vision** on Groq to identify the uploaded object and returns a human-friendly rejection message.
5. **The "Brain" (Reasoning Model):** Filtered results are passed to **Llama-3.1-8b-instant** on Groq, which generates a recommendation. The model is prompted to emit hidden product IDs (`IDs: [prod_X, prod_Y]`) for UI synchronization.
6. **Intent-Product Alignment Layer:** A regex parser extracts the IDs from the LLM response and filters the `products` array so the frontend gallery **only** shows items the LLM actually endorsed.

---

## 3. Technology Stack

| Component | Technology | Reason |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, Vanilla JS | Lightweight, zero-build glassmorphism UI with dynamic product cards. |
| **Charts** | Chart.js | Interactive pie chart with hover-to-see-percentage tooltips on the statistics page. |
| **Backend** | FastAPI, Uvicorn | High-performance async API for ML model inference and agent dispatch. |
| **LLM (Reasoning)** | Llama-3.1-8b-instant (Groq) | Non-reasoning model for instant text generation without `<think>` tag overhead. |
| **LLM (Vision Fallback)** | Llama-3.2-11b-vision-preview (Groq) | Used **only** for OOD rejection to identify non-electronic uploads. |
| **Text Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Ultra-fast 384D text encoding for sub-10ms text search. |
| **Visual Embeddings** | OpenCLIP (`ViT-B-16-SigLIP-256`) | State-of-the-art zero-shot image-text retrieval for the product catalog. |
| **Vector Search** | FAISS (Dual-index: `catalog.index` + `text_catalog.index`) | In-memory, local vector search without external cloud DB dependencies. |
| **Environment** | python-dotenv (`.env` / `api_key.env`) | Secure API key management. |

---

## 4. Project Structure

```
AI_Agent_for_Commerce_Website/
├── backend/
│   ├── main.py                  # FastAPI app, chat endpoint, statistics API, OOD handler
│   ├── search_engine.py         # CatalogSearchEngine: Dual-index, RRF, Zero-Shot Gate
│   └── api_key.env              # Groq API key (gitignored)
├── frontend/
│   ├── index.html               # Main search UI with glassmorphism design
│   ├── style.css                # Global styles, animations, product cards
│   ├── script.js                # Time-aware greeting, fetch API, dynamic UI rendering
│   ├── statistics.html          # Category distribution dashboard
│   └── statistics.js            # Chart.js pie chart with hover tooltips
├── src/
│   └── data_preprocess/
│       ├── data_cleaning.py     # Raw CSV → cleaned JSONL with category assignment
│       ├── download_images.py   # Download product images from Amazon URLs
│       ├── build_index.py       # Build 768D OpenCLIP FAISS index (catalog.index)
│       └── build_text_index.py  # Build 384D MiniLM FAISS index (text_catalog.index)
├── data/
│   ├── raw/                     # Original Kaggle CSV
│   └── processed/               # cleaned_catalog_with_images.jsonl, catalog.index, text_catalog.index
├── documentation/
│   ├── initial_plan.md          # Original architecture proposal
│   └── updated_plan.md          # This file
└── .env.example                 # Template for API keys
```

---

## 5. API Endpoints

### `GET /` — Health Check
Returns `{"status": "ok", "message": "Aura Backend is running!"}`.

### `GET /api/statistics` — Catalog Analytics
Returns the category distribution of the entire product catalog as a JSON object for the pie chart.
```json
{"categories": {"Other Electronics": 5420, "Headphones": 3100, "TV & Display": 2800, ...}}
```

### `POST /api/chat` — The Core Agent Endpoint
Accepts multipart form data with a `message` (text) and optional `image` (file upload).

**Request:**
```
POST /api/chat
Content-Type: multipart/form-data

message: "Find me wireless earbuds"
image: (optional file)
```

**Response (Success):**
```json
{
  "agent_response": "According to the information you provide, these are the products that you may be interested: ...",
  "products": [
    {"rank": 1, "score": 0.95, "product_id": "prod_4133", "title": "Apple AirPods Pro 2nd Gen", "category": "Headphones", "price": "278.0", "image_url": "https://...", "url": "https://..."}
  ]
}
```

**Response (OOD Rejection):**
```json
{
  "agent_response": "According to the image you provided, you are likely looking for a wooden table. I'm sorry, but this appears to be outside of our catalog...",
  "products": []
}
```

---

## 6. Key Design Decisions

### A. Dual-Index over Single-Index
Using a single OpenCLIP index for both text and image queries introduced ~500ms latency for every text-only search (because OpenCLIP is a heavyweight vision model). By decoupling text search to the ultra-lightweight MiniLM encoder (22.7M params vs. 150M), text queries now resolve in under 10ms.

### B. Distractor Gate over Strict Category Matching
An earlier approach attempted to force the Zero-Shot classification label (e.g., "Monitors") to exactly match the catalog category string (e.g., "TV & Display"). This caused false negatives for semantically overlapping categories. The current approach uses a binary Distractor Gate: if the image is clearly non-electronic (Fruit, Animal, Furniture, etc.), it is rejected. Otherwise, FAISS returns the closest visual neighbors regardless of label strings.

### C. Intent-Product Alignment Layer
The LLM is instructed to emit hidden product IDs at the end of its response. A regex parser extracts these IDs and uses them to filter the `products` array before it reaches the frontend, ensuring the visual product gallery is perfectly synchronized with the text recommendation. This prevents the common RAG failure mode where FAISS returns 5 items but the LLM only endorses 2.

### D. Cloud Vision Fallback (Selective)
The Llama 3.2 Vision API is called **only** when local retrieval fails (zero valid products). This "pay the latency tax only on failure" approach keeps the happy path fast while providing a graceful, human-friendly rejection for out-of-distribution images.

### E. Llama-3.1-8b-instant over DeepSeek-R1
The original DeepSeek-R1-Distill-Llama-70B model was both decommissioned by Groq and introduced significant latency due to its Chain-of-Thought `<think>` tags. Switching to a non-reasoning model eliminated the thinking delay entirely.

---

## 7. Shortcomings & Known Limitations

### 1. The "Semantic Gap" (Visual Blindness)
The Reasoning Model (Llama 3.1) has no way to verify if the retrieval results are actually correct. If OpenCLIP returns visually similar but semantically wrong items, the LLM will fabricate a rationale.

**Mitigation:** The `valid_cats` whitelist and the Distractor Gate reduce—but do not eliminate—this risk.

### 2. Dependency on Metadata Quality
Scraped Amazon data is notoriously "dirty." If a product title is just "Product X-123" and the description is empty, the LLM receives zero context about that item, even if the image match was 100% perfect.

### 3. Category Catchall
The `data_cleaning.py` keyword-based classifier assigns any unrecognized product to `"Other Electronics"`. This means non-electronic items (hand sanitizer, baby ointment) in the raw dataset are classified as electronics. The `valid_cats` whitelist in `main.py` currently includes `"Other Electronics"` only if explicitly added—otherwise these items are filtered out.

### 4. Single-Turn Conversations
The current `/api/chat` endpoint does not maintain conversation history. Each request is stateless. Multi-turn context (e.g., "Show me that second one in red") is not supported.

---

## 8. Setup & Reproduction

### Prerequisites
- Python 3.10+ with conda environment (`aura`)
- Dependencies: `fastapi`, `uvicorn`, `python-dotenv`, `groq`, `faiss-cpu`, `open_clip_torch`, `sentence-transformers`, `torch`, `pandas`, `Pillow`

### Steps
1. **Clean the raw data:**
   ```bash
   python src/data_preprocess/data_cleaning.py
   ```
2. **Download product images:**
   ```bash
   python src/data_preprocess/download_images.py
   ```
3. **Build the Visual FAISS index (768D):**
   ```bash
   python src/data_preprocess/build_index.py
   ```
4. **Build the Text FAISS index (384D):**
   ```bash
   python src/data_preprocess/build_text_index.py
   ```
5. **Set your API key** in `.env` or `backend/api_key.env`:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
6. **Start the backend:**
   ```bash
   cd backend && uvicorn main:app --reload
   ```
7. **Open the frontend:**
   Open `frontend/index.html` in your browser.

---

## 9. Next Steps

- [ ] **Streaming (SSE):** Implement Server-Sent Events in FastAPI to stream the LLM response token-by-token for a more responsive UI.
- [ ] **Multi-Turn Context:** Add conversation history to the `/api/chat` endpoint for follow-up queries.
- [ ] **Production Deployment:** Containerize with Docker and deploy to a cloud provider.
- [ ] **Evaluation:** Build an automated test suite with known image→product pairs to measure retrieval accuracy (Recall@K, NDCG).