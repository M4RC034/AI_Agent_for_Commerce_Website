import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_text_index():
    # 1. Paths relative to project root (nested in src/data_preprocess.py/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    catalog_path = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_catalog_with_images.jsonl')
    index_output_path = os.path.join(BASE_DIR, 'data', 'processed', 'text_catalog.index')

    print(f"Loading metadata from {catalog_path}...")
    df = pd.read_json(catalog_path, orient='records', lines=True)

    # Prepare sentences for embedding (mixing title and category)
    sentences = []
    for _, row in df.iterrows():
        title = str(row.get('product_title', ''))
        category = str(row.get('product_category', ''))
        sentences.append(f"{category} - {title}")

    # 2. Initialize MiniLM model
    print("Initializing sentence-transformers (all-MiniLM-L6-v2) on CPU/GPU...")
    # By default, sentence-transformers uses the fastest available device (CUDA, MPS, or CPU)
    # The output is 384 dimensions.
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Batch encode the entire catalog
    print(f"Encoding {len(sentences)} items into 384-dimensional space...")
    
    # We set normalize_embeddings=True to use Cosine Similarity natively
    embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    embeddings = embeddings.astype('float32') # FAISS requires float32

    # 4. Create and populate FAISS Index
    dimension = 384
    # IndexFlatIP uses Inner Product. Since vectors are normalized, IP === Cosine Similarity.
    faiss_text_index = faiss.IndexFlatIP(dimension)
    
    print(f"Building FAISS text index (D={dimension})...")
    faiss_text_index.add(embeddings)
    
    # 5. Save the 384D Index to disk
    faiss.write_index(faiss_text_index, index_output_path)
    print(f"Successfully saved Dual-Index Component to {index_output_path}")

if __name__ == "__main__":
    build_text_index()
