# Al — AI-Powered Commerce Agent

An AI-powered multimodal shopping assistant for an electronics commerce website, inspired by [Amazon Rufus](https://www.aboutamazon.com/news/retail/amazon-rufus). A single unified agent handles **general conversation**, **text-based product recommendations**, and **image-based product search** — all within one endpoint.

**Author:** Marco Wang

🚀 **Live Demo:** [huggingface.co/spaces/Marc034/al-commerce-agent](https://huggingface.co/spaces/Marc034/al-commerce-agent)

---

## Features

| Feature | Example | How It Works |
|---|---|---|
| **General Conversation** | "What's your name?", "What can you do?" | Intent detection bypasses retrieval; LLM responds conversationally |
| **Text-Based Recommendation** | "Recommend me wireless earbuds under $100" | MiniLM encoder → FAISS text index → Cross-Encoder re-rank → LLM reasoning |
| **Image-Based Search** | Upload a photo of headphones | OpenCLIP encoder → FAISS visual index → Distractor Gate → LLM reasoning |
| **Hybrid (Text + Image)** | Upload photo + "find me something like this but cheaper" | Both encoders → Reciprocal Rank Fusion → Cross-Encoder re-rank → LLM reasoning |
| **Dynamic Price Filtering** | "Laptops under $500" | Regex extracts price constraint → FAISS overfetch → numeric post-filter |
| **Attribute Comparison** | "Which of these has the most storage?" | Metadata hints injected into RAG context → LLM performs MAX/MIN reasoning |
| **Explainable AI Badges** | Product card shows "💰 Budget Pick" | LLM generates reasoning badges → backend parses JSON → frontend renders glassmorphic pills |
| **Catalog Analytics** | Statistics dashboard | `/api/statistics` → Chart.js interactive pie chart |
| **Search History** | Persistent query history panel | `localStorage`-based, per-user, survives page refresh |

All use cases are served by a **single `/api/chat` endpoint** — no separate agents.

---

## Architecture

```
User Input (text / image / both)
        │
        ▼
┌─────────────────────────┐
│    Intent Detection     │ ──── General chat? ──→ LLM (conversational mode)
└──────────┬──────────────┘
           │ Product query
           ▼
┌─────────────────────────┐
│  Dynamic Pre-Filter     │ ──── "under $100" → max_price = 100
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Smart Router           │
│ ┌─────────┬───────────┐ │
│ │ MiniLM  │ OpenCLIP  │ │
│ │ (384D)  │ (768D)    │ │
│ └────┬────┴─────┬─────┘ │
│      │   FAISS  │       │
│      ▼          ▼       │
│ text_catalog  catalog   │
│   .index      .index    │
└──────────┬──────────────┘
           │ Late Fusion (RRF) + Price Filter
           ▼
┌─────────────────────────┐
│  Cross-Encoder Re-Rank  │ ──── ms-marco-MiniLM-L-6-v2 (top 15 → top 5)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Category Safety Net    │ ──── Out-of-domain? ──→ Llama 3.2 Vision (OOD rejection)
└──────────┬──────────────┘
           │ Valid products
           ▼
┌─────────────────────────┐
│  Llama 3.1 (Groq)       │ ──→ Generates recommendation + product IDs + XAI Badges
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Intent-Product Alignment│ ──→ Syncs LLM text with product gallery
└─────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JS | Lightweight, zero-build glassmorphism UI |
| **Charts** | Chart.js | Interactive pie chart with hover tooltips |
| **Backend** | FastAPI + Uvicorn | High-performance async API for ML inference |
| **LLM (Reasoning)** | Llama-3.1-8b-instant (Groq) | Fast, non-reasoning model — no `<think>` tag overhead |
| **LLM (Vision Fallback)** | Llama-3.2-11b-vision-preview (Groq) | Used only for OOD rejection on failed image searches |
| **Text Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Ultra-fast 384D encoding for sub-10ms text search |
| **Visual Embeddings** | OpenCLIP (`ViT-B-16-SigLIP-256`) | State-of-the-art zero-shot image-text retrieval |
| **Vector Search** | FAISS (dual-index) | In-memory local vector search, no cloud DB dependency |
| **Re-Ranking** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Two-stage retrieve-and-rerank for precision without latency penalty |
| **Environment** | python-dotenv | Secure API key management via `.env` with Docker quote sanitization |

### Why these choices?

- **Dual-Index over Single-Index:** A single OpenCLIP index added ~500ms latency for every text query. Decoupling text search to the lightweight MiniLM encoder (22.7M vs 150M params) reduced text query latency to under 10ms.
- **Groq over OpenAI/Anthropic:** Free tier with extremely fast inference (sub-second). Ideal for a demo that needs to feel responsive.
- **Vanilla JS over React:** Zero build step, instant deployment as static files, and the UI complexity doesn't warrant a framework.
- **FAISS over Pinecone/Weaviate:** No external service dependencies — the entire system runs locally or on a single container.

---

## Project Structure

```
AI_Agent_for_Commerce_Website/
├── backend/
│   ├── main.py                  # FastAPI app: chat endpoint, statistics API, intent detection
│   ├── search_engine.py         # CatalogSearchEngine: dual-index, RRF, Zero-Shot Gate
│   ├── requirements.txt         # Python dependencies
│   └── api_key.env              # Groq API key (gitignored)
├── frontend/
│   ├── index.html               # Main search UI with glassmorphism design
│   ├── style.css                # Global styles, animations, product cards
│   ├── script.js                # Time-aware greeting, fetch API, dynamic rendering
│   ├── statistics.html          # Category distribution dashboard
│   └── statistics.js            # Chart.js pie chart
├── src/
│   ├── data_cleaning.py     # Raw CSV → cleaned JSONL with category assignment
│   ├── download_images.py   # Download product images from Amazon URLs
│   ├── build_index.py       # Build 768D OpenCLIP FAISS index
│   └── build_text_index.py  # Build 384D MiniLM FAISS index
├── data/
│   ├── raw/                     # Original Kaggle CSV
│   └── processed/               # cleaned_catalog_with_images.jsonl, *.index files
├── documentation/
│   ├── initial_plan.md          # Original architecture proposal
│   └── updated_plan.md          # Current architecture documentation
├── .env.example                 # Template for API keys
└── .gitignore
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier available)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/M4RC034/AI_Agent_for_Commerce_Website.git
cd AI_Agent_for_Commerce_Website
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_your_key_here
```
Make sure to put you `api_key.env` inside the `backend` folder.

### 3. Prepare the Data (first time only)

Download the [Amazon Products Sales Dataset 42K+ Items - 2025](https://www.kaggle.com/datasets/ikramshah512/amazon-products-sales-dataset-42k-items-2025) and place the `amazon_products_sales_data_uncleaned.csv` file in the `data/raw/` folder.

```bash
# Clean the raw Amazon CSV
python src/data_preprocess/data_cleaning.py

# Download product images
python src/data_preprocess/download_images.py

# Build the Visual FAISS index (768D, OpenCLIP)
python src/data_preprocess/build_index.py

# Build the Text FAISS index (384D, MiniLM)
python src/data_preprocess/build_text_index.py
```

### 4. Docker (How to access this interface locally)

```bash
docker build -t al-agent .
docker run -p 8000:8000 --env-file backend/api_key.env al-agent
```
Then go to http://localhost:8000 in your browser.
---

## Key Design Decisions

1. **Intent Detection Gate:** Conversational queries (greetings, identity questions) bypass the retrieval pipeline entirely and go straight to the LLM. Product signals (keywords like "recommend", "laptop") override greeting patterns, so "Hi, find me a laptop" still routes to product search.

2. **Distractor Gate over Category Matching:** Instead of demanding exact category string matches between OpenCLIP labels and catalog labels (which caused false negatives), a binary distractor gate rejects clearly non-electronic images and lets FAISS return the closest visual neighbors for everything else.

3. **Intent-Product Alignment Layer:** The LLM emits hidden product IDs at the end of its response. A regex parser extracts these and filters the product array so the frontend gallery shows only items the LLM actually endorsed.

4. **Cloud Vision Fallback (Selective):** Llama 3.2 Vision is called only when local retrieval fails (zero valid products), keeping the happy path fast.

5. **Client-Side Conversation History:** Multi-turn context is maintained in the browser (no server-side storage or database needed). The frontend sends the last 5 turns as a JSON array with each request, and the backend injects them into the Groq message array. This enables follow-ups like "show me a cheaper one" while keeping the architecture stateless and deployment-simple.

6. **Two-Stage Retrieve and Re-Rank Pipeline:** To solve the inherent precision issues of Bi-encoder vector search without destroying responsiveness, the architecture uses a two-stage pattern. Phase 1 leverages FAISS and MiniLM for fast, high-recall retrieval (casting a wider net of `k=15` candidates). Phase 2 uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to perform deep semantic re-ranking, efficiently recalculating the relevance permutations of the user's query against the product titles to accurately slice the top 5. This dramatically bridges the 'Semantic Gap'. A strict **Visual Safety Net** prevents purely visual OpenCLIP queries from hitting the text-based Cross-Encoder, actively balancing computational latency against maximum precision.

7. **Explainable AI (XAI) Badges:** To build user trust and transparency, the system instructs the LLM to generate a short reasoning badge (e.g., `✨ Visual Match`, `💰 Budget Pick`, `🏆 Top Choice`) for each recommended product. The backend extracts these via a robust regex parser with markdown-sanitization fallbacks (`json.loads` with backtick stripping), maps them to the product objects, and assigns a safe `🎯 Recommended` default if the LLM omits or malforms the JSON. The frontend renders these as glassmorphic pill overlays (`backdrop-filter: blur`) positioned absolutely on each product card image, giving users immediate visual insight into *why* the AI selected each item.

8. **Dynamic Price Pre-Filtering:** Before any vector search occurs, a regex parser extracts price constraints from the user's natural language (e.g., "under $100", "cheaper than $50"). The extracted `max_price` is passed to the search engine, which uses an **overfetch-then-filter** paradigm: FAISS retrieves `k×10` candidates, then `_filter_and_rerank` numerically eliminates items exceeding the budget and slices the true top-K. This bolts hard metadata constraints onto a pure vector engine.

9. **Attribute Extraction Hints:** To enable accurate product comparisons (e.g., "which has the most storage?"), the backend injects structured `HINT` lines into the RAG context string (e.g., `HINT: Result 1 has 12TB Storage`). This gives the 8B LLM explicit pre-extracted metadata to perform MAX/MIN reasoning without needing to parse raw product titles.

10. **API Key Sanitization:** The Groq API key loader applies `.strip().strip('"').strip("'")` to handle edge cases where Docker `--env-file` passes literal quote characters around the key value, preventing silent `401 Unauthorized` failures in containerized deployments.

---

## Known Limitations

- **Context Window:** Conversation history is capped at 5 turns (10 messages) to stay within token limits. Very long conversations will lose early context.
- **Semantic Gap (Partially Mitigated):** The reasoning model (Llama 3.1) cannot visually verify retrieval results. However, the Cross-Encoder re-ranking stage (Design Decision #6) now deeply re-scores the top 15 FAISS candidates against the user's actual query before they reach the LLM, dramatically reducing semantically wrong items from slipping through. The remaining gap exists for **image-only queries**, which bypass the text-based Cross-Encoder entirely by design.
- **Unweighted Late Fusion Conflicts (Partially Mitigated):** When using hybrid search (image + text), the RRF algorithm still equally weights both indexes. However, the Cross-Encoder now acts as a **second-pass semantic correction layer** after fusion — even if a gaming mouse initially outranks keyboards in the RRF math, the Cross-Encoder will re-score `["mechanical keyboards", "Razer Gaming Mouse"]` far lower than `["mechanical keyboards", "Corsair K70 Keyboard"]`, pushing correct items to the top. A production-grade resolution would still benefit from dynamically weighting the RRF scores based on primary intent classification.
---

## API Documentation

FastAPI auto-generates interactive API documentation. Once the backend is running:

| URL | Description |
|---|---|
| [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) | **Swagger UI** — Interactive docs where you can test all endpoints directly in the browser |
| [`http://127.0.0.1:8000/redoc`](http://127.0.0.1:8000/redoc) | **ReDoc** — Clean, read-only API reference |

### Endpoints

#### `GET /` — Health Check

Returns server status.

```json
{"status": "ok", "message": "Al Backend is running!"}
```

#### `GET /api/statistics` — Catalog Analytics

Returns category distribution for the statistics dashboard.

```json
{
  "categories": {
    "Other Electronics": 5420,
    "Headphones": 3100,
    "TV & Display": 2800,
    "...": "..."
  }
}
```

#### `POST /api/chat` — The Core Agent Endpoint

The unified endpoint that handles all three use cases: general conversation, text-based recommendations, and image-based search.

**Request** (multipart form data):

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | Yes | User's text query |
| `image` | file | No | Product image (JPG, PNG, WebP, HEIC) |
| `history_json` | string | No | JSON array of previous conversation turns (default: `"[]"`) |

**Response (product recommendation):**

```json
{
  "agent_response": "According to the information you provide, these are the products...",
  "products": [
    {
      "rank": 1,
      "score": 0.95,
      "product_id": "prod_4133",
      "title": "Apple AirPods Pro 2nd Gen",
      "category": "Headphones",
      "price": "278.0",
      "image_url": "https://...",
      "url": "https://...",
      "badge": "🏆 Top Choice"
    }
  ]
}
```

**Response (general conversation):**

```json
{
  "agent_response": "I'm Al, your AI shopping assistant! I can help you find electronics...",
  "products": []
}
```

**Response (out-of-domain image):**

```json
{
  "agent_response": "According to the information you provided, you are likely looking for a wooden table. I'm sorry, but this appears to be outside of our catalog...",
  "products": []
}
```

---

## License

This project was built as a take-home exercise by Palona AI.
