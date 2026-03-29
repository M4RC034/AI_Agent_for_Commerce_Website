import os
import torch
import open_clip
import faiss
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

def build_faiss_index(input_jsonl, index_output_path):
    print(f"Loading catalog from {input_jsonl}...")
    df = pd.read_json(input_jsonl, orient='records', lines=True)
    
    # Determine the device (Use MPS for Apple Silicon, CUDA for Nvidia, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    # Load the powerful SigLIP model
    print("Loading OpenCLIP ViT-B-16-SigLIP model...")
    model_name = 'ViT-B-16-SigLIP-256'
    pretrained = 'webli'
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval() # Set model to evaluation mode
    
    tokenizer = open_clip.get_tokenizer(model_name)
    
    embeddings = []
    
    print(f"Generating embeddings for {len(df)} products...")
    # Wrap with no_grad to drastically reduce memory usage during inference
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
            img_path = row.get('local_image_path')
            emb = None
            
            # 1. Attempt Visual Encoding
            if img_path and isinstance(img_path, str) and os.path.exists(img_path):
                try:
                    # Load, preprocess, and add batch dimension
                    image = Image.open(img_path).convert('RGB')
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    
                    # Encode Image
                    image_features = model.encode_image(image_input)
                    # Normalize the embedding for Cosine Similarity (Inner Product)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    emb = image_features.cpu().numpy().astype('float32')
                except (UnidentifiedImageError, OSError):
                    # Catch corrupt images or unreadable files
                    emb = None
            
            # 2. Fallback to Semantic Text Encoding if image was missing or corrupt
            if emb is None:
                # Build a rich "context" string for the fallback
                title = str(row.get('product_title', ''))
                category = str(row.get('product_category', ''))
                fallback_text = f"Product: {title} Category: {category}"
                
                text_input = tokenizer([fallback_text]).to(device)
                text_features = model.encode_text(text_input)
                
                # Normalize the embedding
                text_features /= text_features.norm(dim=-1, keepdim=True)
                emb = text_features.cpu().numpy().astype('float32')
                
            embeddings.append(emb[0])
            
    # Stack all embeddings together into a single [N, Dimension] array
    embeddings_matrix = np.vstack(embeddings)
    embedding_dim = embeddings_matrix.shape[1]
    
    print(f"Generated {len(embeddings)} embeddings of dimension {embedding_dim}")
    
    # 3. Build & Save the FAISS Index
    print("Building FAISS index (IndexFlatIP for Cosine Similarity)...")
    # Using IndexFlatIP because our vectors are normalized, making Inner Product equal Cosine Similarity
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_matrix)
    
    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(index, index_output_path)
    print(f"Success! FAISS index saved to {index_output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(BASE_DIR, "data", "processed", "cleaned_catalog_with_images.jsonl")
    index_output = os.path.join(BASE_DIR, "data", "processed", "catalog.index")
    
    if os.path.exists(input_file):
        build_faiss_index(input_file, index_output)
    else:
        print(f"Error: {input_file} not found. Did you successfully download the images?")
