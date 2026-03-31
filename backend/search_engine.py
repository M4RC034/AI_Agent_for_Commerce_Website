import os
import re
import torch
import open_clip
import faiss
import pandas as pd
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class CatalogSearchEngine:
    def __init__(self, index_path, catalog_path):
        print(f"Loading FAISS Image index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        # Load the new text index
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        text_index_path = os.path.join(BASE_DIR, "data", "processed", "text_catalog.index")
        print(f"Loading FAISS Text index from {text_index_path}...")
        try:
            self.text_index = faiss.read_index(text_index_path)
        except Exception as e:
            print("WARNING: Text index not found. Text search will fail.")
            self.text_index = None

        print(f"Loading catalog metadata from {catalog_path}...")
        self.df = pd.read_json(catalog_path, orient='records', lines=True)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        print(f"Initializing OpenCLIP on {self.device}...")
        model_name = 'ViT-B-16-SigLIP-256'
        pretrained = 'webli'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        print("Initializing SentenceTransformer (all-MiniLM-L6-v2)...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

        # --- Zero-Shot Classification Gate (for Images only) ---
        print("Pre-computing Zero-Shot Category Vectors...")
        self.valid_categories = ['Laptops', 'Phones', 'Headphones', 'Chargers & Cables', 'Cameras', 'Storage', 'Smart Home', 'TV & Display', 'Power & Batteries', 'Networking', 'Wearables', 'Speakers', 'Printers & Scanners', 'Gaming']
        self.distractor_categories = ['Fruit', 'Food', 'Animal', 'Furniture', 'Clothing', 'Vehicle', 'Plant', 'Weapon', 'Person', 'Toy', 'Hardware Tool']
        self.all_category_names = self.valid_categories + self.distractor_categories
        prompt_templates = [f"a photo of a {cat.lower()}" for cat in self.all_category_names]
        with torch.no_grad():
            text_tokens = self.tokenizer(prompt_templates).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            self.category_vectors = self._normalize(text_features)
            
        print("Dual-Index Search Engine Ready!")

    def _normalize(self, features):
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype('float32')

    def search_by_text(self, text_query, k=5, max_price=None):
        """Uses MiniLM to search the dedicated text index."""
        if not self.text_index:
            print("ERROR: Dual Index not available for text search.")
            return []
            
        # Encode with SentenceTransformers and normalize (required for Inner Product)
        query_vector = self.text_model.encode([text_query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        
        # Over-fetch if max_price is provided to ensure we have enough results after filtering
        fetch_k = k * 10 if max_price is not None else k
        distances, indices = self.text_index.search(query_vector, fetch_k)
        
        raw_results = self._fetch_results(indices[0], distances[0])
        return self._filter_and_rerank(raw_results, max_price, k)

    def search_by_image(self, image_input: Image.Image, k=5, max_price=None):
        """Uses OpenCLIP to search the image index."""
        with torch.no_grad():
            image_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            query_vector = self._normalize(image_features)
            
            # Zero-Shot Gate Check
            similarity = (query_vector @ self.category_vectors.T)
            best_cat_idx = np.argmax(similarity)
            best_category = self.all_category_names[best_cat_idx]
            
            if best_category in self.distractor_categories:
                return [{"rank": 1, "score": 0.0, "product_id": "NONE", "title": "OUT_OF_BOUNDS_IMAGE", "category": "Error", "price": "N/A", "image_url": "", "url": "", "description": f"Internal Note: The uploaded image was classified as a '{best_category}', not an electronic device. Refused to search catalog."}]
                
        # We now rely solely on the Distractor Gate above to kill completely OOD objects (Furniture, etc.)
        # If the object is generally electronic, we simply return the closest FAISS neighbors 
        # instead of incorrectly demanding the OpenCLIP zero-shot `best_category` string perfectly 
        # match the FAISS catalog string (which caused False Negatives for fuzzy concepts like Monitors vs TVs).
        
        fetch_k = k * 10 if max_price is not None else k
        distances, indices = self.index.search(query_vector, fetch_k)
        raw_results = self._fetch_results(indices[0], distances[0])
        return self._filter_and_rerank(raw_results, max_price, k)

    def hybrid_search(self, text_query, image_input, k=5, max_price=None):
        """Performs Late Fusion using Reciprocal Rank Fusion (RRF)."""
        fetch_k = k * 10 if max_price is not None else 10
        image_results = self.search_by_image(image_input, k=fetch_k) # Over-fetch for merging
        if image_results and image_results[0].get('product_id') == 'NONE':
            return image_results # Fail fast on OOD gate
            
        text_results = self.search_by_text(text_query, k=fetch_k)
        
        # RRF Scoring Map
        rrf_scores = {}
        items_map = {}
        
        # Merge lists, ranking parameter k=60 is standard
        for rank, item in enumerate(image_results):
            pid = item['product_id']
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (rank + 1 + 60))
            items_map[pid] = item
            
        for rank, item in enumerate(text_results):
            pid = item['product_id']
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (rank + 1 + 60))
            items_map[pid] = item
            
        # Sort by best combined RRF score
        sorted_pids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Format results
        combined = []
        for pid in sorted_pids:
            item = items_map[pid]
            item['score'] = round(rrf_scores[pid] * 100, 4) # Multiply by 100 for readability
            combined.append(item)
            
        return self._filter_and_rerank(combined, max_price, k)

    def _fetch_results(self, indices, scores):
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx == -1: continue 
            row = self.df.iloc[idx].fillna("")
            results.append({
                "rank": rank + 1,
                "score": round(float(score), 4),
                "product_id": str(row.get('product_id', '')),
                "title": str(row.get('product_title', '')),
                "category": str(row.get('product_category', '')),
                "price": str(row.get('discounted_price', row.get('original_price', 'N/A'))),
                "image_url": str(row.get('product_image_url', '')),
                "url": str(row.get('product_page_url', ''))
            })
        return results

    def _filter_and_rerank(self, results, max_price, limit):
        if max_price is None:
            final_list = results[:limit]
        else:
            filtered = []
            for item in results:
                price_str = item.get('price', '')
                if price_str == 'N/A':
                    continue
                try:
                    num_str = re.sub(r'[^\d.]', '', str(price_str))
                    if num_str:
                        val = float(num_str)
                        if val <= max_price:
                            filtered.append(item)
                except Exception:
                    pass
            final_list = filtered[:limit]
            
        # Write sequential ranks
        for i, res in enumerate(final_list):
            res['rank'] = i + 1
            
        return final_list
