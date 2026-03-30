# Multimodal AI Commerce Agent

This is a ready-to-use AI shopping assistant designed for modern commerce. Inspired by Amazon Rufus, Aura uses a "Lean" architecture to handle both text queries and visual inputs efficiently through a single conversational agent.

## Technical Highlights
* **The Brain Orchestrator:** A ReAct Agent (powered by an LLM) that acts as the Dispatcher, handling conversations and deciding when to search the catalog.
* **The Worker Index:** A fast, local tool powered by OpenCLIP and FAISS for vector retrieval, keeping the project performant and self-contained without cloud dependencies.
* **Multimodal Tool-Use:** The agent autonomously handles text-to-product, image-to-product, and complex intent-based queries.

## Technology Stack

| Component | Technology | Reason |
| :--- | :--- | :--- |
| **Frontend** | Next.js 14, Tailwind CSS | Fast, responsive UI capable of displaying chat and product cards. |
| **Backend** | FastAPI (Python) | High-performance API serving for ML model inference and agent dispatch. |
| **LLM Orchestrator** | GPT-4o | Industry-leading reasoning for tool calling and multimodal analysis. |
| **Local Index** | FAISS | Ultra-low latency, in-memory local vector search without external cloud DBs. |
| **Embeddings** | OpenCLIP (`ViT-B-16-SigLIP`) | State-of-the-art zero-shot image-text retrieval for our product catalog. |

## Architecture & Logic

### 1. Data Engineering Pipeline
Powered by the **Amazon Products Sales Dataset 42K+ Items - 2025** (from Kaggle).
* **Cleaning & Feature Engineering:** Processed via `data_cleaning.py` to extract clean titles, prices, and features.
* **Storage:** Cleaned catalog is saved to a `.jsonl` file for the backend to ingest.

### 2. Pre-computing the Local Index
A one-time script encodes the product images using OpenCLIP and saves them locally into a FAISS index (`catalog.index`).

### 3. The Agent Dispatch Logic (The "Smart" Part)
When a user interacts, GPT-4o decides which "mode" of the tool to use:
* **Scenario 1: Text-to-Product:** User asks for "red running shoes." The LLM calls the text search tool. OpenCLIP encodes the text $\rightarrow$ searches FAISS.
* **Scenario 2: Image-to-Product:** User uploads a photo of a blue watch. The LLM calls the visual search tool. OpenCLIP encodes the image $\rightarrow$ searches FAISS.
* **Scenario 3: The "Complex" Query:** User uploads a photo of a blue watch and asks "Do you have this, but with a leather strap?" The LLM understands the specific attributes and generates a precise text query: `search_catalog(query="blue watch silver dial leather strap")`.

## API & Interface Design

### The `/chat` Endpoint
A single conversational endpoint handles everything, satisfying the API documentation requirement.

**Request:**
```json
// POST /api/chat
{
  "message": "I like this style, what else do you have?",
  "image_base64": "optional_string",
  "history": []
}
```

**Response:**
Returns both the agent's conversational response and the structured product data for frontend rendering.
```json
{
  "agent_response": "I see you're looking for minimalist watches! Based on that image, I found these 3 matches in our catalog:",
  "products": [
    {"id": 101, "name": "Seiko 5 Sport", "price": 199.99, "img": "...url"},
    {"id": 105, "name": "Timex Weekender", "price": 45.00, "img": "...url"}
  ]
}
```

## Setup & Implementation Timeline

1.  **Day 1:**
    * Clean the Kaggle CSV using `data_cleaning.py` and download ~100 sample images.
    * Run a script to create the `catalog.index` using OpenCLIP and FAISS.
2.  **Day 2:**
    * Build the FastAPI backend with one Tool-Enabled Agent.
    * Create a basic Next.js/React chat UI that displays product cards.
3.  **Day 3:**
    * Write the `README.md` and complete API documentation.