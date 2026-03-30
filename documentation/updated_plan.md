# Architecture Update: Adopting "Vision-via-Retrieval"

## 1. Overview of the Pivot
In our initial plan (`initial_plan.md`), the architecture outlined using a single Multi-modal LLM (e.g., GPT-4o or Claude 3.5 Sonnet) as the primary engine to process both text constraints and direct image uploads. 

After further review, we have pivoted to a **"Vision-via-Retrieval"** RAG (Retrieval-Augmented Generation) pipeline. The core difference is that the final Reasoning Agent (LLM) is entirely **blind** to the image itself. Instead, a specialized Machine Learning pipeline evaluates the image locally and injects explicit product matches as text context into the LLM prompt.

### The New Workflow:
1. **The Request:** User uploads a product photo (e.g., a blue watch) to the `/api/chat` endpoint.
2. **The "Eyes" (Local Indexing):** The FastAPI backend uses **OpenCLIP** to encode the image into a 768-dimensional space.
3. **The Retrieval:** **FAISS** inherently runs a vector similarity search across the entire pre-computed `catalog.index` and returns the top 5 closest matched product entries.
4. **Context Injection:** The raw metadata of these 5 products is formatted into a text string.
5. **The "Brain" (Reasoning Model):** We prompt an elite open-weights Reasoning model (DeepSeek-R1) with this text context alongside the user's initial message. The AI synthesizes the best recommendation.

---

## 2. Why We Decided to Make This Change

We selected this pivot over a pure Multi-modal LLM API call for three huge reasons highly relevant to an engineering take-home exercise:

### A. Showcasing Real ML Engineering (Decoupling)
Sending an image blindly to the OpenAI API proves you can write an API wrapper. Building a decoupled "Vision-via-Retrieval" pipeline proves you understand how to build resilient, modular, and controllable AI systems. This showcases core competencies in Data Engineering, Vector Databases, Embedding Models, and Advanced RAG implementation.

### B. Speed & Latency Optimization
Cloud-based Vision LLMs (like GPT-4 Vision) are notoriously slow—often taking 3–6 seconds just to process image tokens. By delegating the "vision" computationally to a local FAISS index (resolves in `<10ms`), and passing only text to **Groq's** LPU infrastructure, the entire recommendation pipeline feels instantaneous to the end-user. 

### C. 100% Free and Readily Replicable 
Running GPT-4o scale applications costs real money per token. By operating embeddings locally and using subsidized developer tiers for open-weight models, we deliver state-of-the-art capability for **$0**, making it exceptionally easy for the reviewer to test and deploy the codebase locally without needing enterprise billing accounts.

---

## 3. Why We Chose This Specific Tech Stack

### OpenCLIP (`ViT-B-16-SigLIP`) instead of OpenAI Embeddings
SigLIP is arguably the strongest open-source model available for mapping textual intent and image pixels into the exact same vector space. Its performance in zero-shot image-text retrieval is incredibly precise. Furthermore, if a product URL 404s (image missing), OpenCLIP effortlessly falls back to embedding the `product_title` directly, preventing product loss.

### FAISS (Local) instead of Pinecone / Qdrant (Cloud)
Our requirement clearly specified that search handles a *predefined catalog* (42K items). A catalog of this size requires roughly 130MB of memory. Maintaining a live network connection to cloud providers like Pinecone for something that sits perfectly in local memory is massive over-engineering. FAISS is perfectly suited to handle this in-memory securely and fast.

### Groq & DeepSeek-R1 instead of GPT-4o
DeepSeek-R1 (specifically the Llama distill versions) currently rivals elite closed-source models in chain-of-thought (CoT) reasoning. Because the "Brain" model relies heavily on complex logic to filter down the FAISS results (e.g., *"User wants a watch like the photo but needs a leather strap instead"*), utilizing DeepSeek's advanced reasoning capabilities hosted on Groq's lightning-fast hardware is simply the best architectural decision.

## 4. Short comings of this approach
### 1. The "Semantic Gap" (Visual Blindness)
The biggest risk is that the Reasoning Model (DeepSeek) has no way to verify if the retrieval results are actually correct.

The Issue: If OpenCLIP returns a "Blue Yoga Mat" instead of a "Blue Sports Shirt" because the visual vectors were close in color/texture, the LLM will see the text "Blue Yoga Mat" and try to justify why it's a good recommendation.

The Impact: Without the ability to cross-reference the pixels against the text, the agent might confidently "hallucinate" a rationale for a visually incorrect match.

### 2. Dependency on Metadata Quality
In this architecture, your system is only as smart as your CSV/JSONL file.

The Issue: Scraped Amazon data is notoriously "dirty." If a product title is just "Product X-123" and the description is empty, the LLM receives zero context about that item, even if the image match was 100% perfect.

The Impact: A native multimodal model can see a "Seiko watch" in a photo even if the metadata is missing the brand name. Your "blind" LLM cannot.