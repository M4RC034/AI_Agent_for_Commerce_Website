import os
import faiss
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image

class CatalogSearchEngine:
    def __init__(self, index_path, catalog_path):
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading catalog metadata from {catalog_path}...")
        self.df = pd.read_json(catalog_path, orient='records', lines=True)
        
        # Initialize the OpenCLIP Model
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
        print("Search Engine Ready!")

    def _normalize(self, features):
        """Normalizes a tensor to unit length for Cosine Similarity inside FAISS"""
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype('float32')

    def search_by_text(self, text_query, k=5):
        """Encodes text and performs vector search against the FAISS index."""
        with torch.no_grad():
            text_input = self.tokenizer([text_query]).to(self.device)
            text_features = self.model.encode_text(text_input)
            query_vector = self._normalize(text_features)
            
        distances, indices = self.index.search(query_vector, k)
        return self._fetch_results(indices[0], distances[0])

    def search_by_image(self, image_input: Image.Image, k=5):
        """Encodes a PIL image and performs vector search against the FAISS index."""
        with torch.no_grad():
            image_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            query_vector = self._normalize(image_features)
            
        distances, indices = self.index.search(query_vector, k)
        return self._fetch_results(indices[0], distances[0])

    def _fetch_results(self, indices, scores):
        """Maps FAISS output indices back to product metadata dictionaries."""
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx == -1: # FAISS returns -1 if not enough results exist
                continue 
            
            # Fetch the entire row as a string/dict
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
