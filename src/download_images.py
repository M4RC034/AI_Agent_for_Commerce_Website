import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_image(url, save_path, timeout=5):
    """
    Downloads an image from a URL.
    Returns True if successful, False otherwise.
    """
    if not isinstance(url, str) or not url.startswith('http'):
        return False
        
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        # Check if the request was successful
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except (requests.exceptions.RequestException, Exception):
        pass
    
    return False

def process_catalog_images(input_jsonl, image_dir, max_downloads=None):
    """
    Reads the catalog, downloads images to local directory, 
    and handles URL not found errors.
    """
    print(f"Loading catalog from {input_jsonl}...")
    df = pd.read_json(input_jsonl, orient='records', lines=True)
    
    # We will use the dataframe index as a unique ID if one doesn't exist
    # Let's add an explicit 'id' column if missing
    if 'id' not in df.columns and 'product_id' not in df.columns:
        df['product_id'] = ["prod_" + str(i) for i in range(len(df))]
        
    os.makedirs(image_dir, exist_ok=True)
    
    # Track paths
    local_image_paths = []
    
    # Limit downloads for testing if specified
    total_to_download = len(df) if max_downloads is None else min(len(df), max_downloads)
    
    # We will use ThreadPoolExecutor to download images concurrently for speed
    print(f"Starting download of {total_to_download} images...")
    
    def process_row(row):
        idx, data = row
        if max_downloads is not None and idx >= max_downloads:
            return idx, None
            
        img_url = data.get('product_image_url', '')
        prod_id = data.get('product_id', f"prod_{idx}")
        save_path = os.path.join(image_dir, f"{prod_id}.jpg")
        
        # Skip if already exists
        if os.path.exists(save_path):
            return idx, save_path
            
        success = download_image(img_url, save_path)
        if success:
            return idx, save_path
        else:
            return idx, None

    # Use 10 concurrent threads to avoid saturating network/Amazon rate limits
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks
        futures = {executor.submit(process_row, row): row[0] for row in df.iterrows()}
        
        # Process as they complete with a progress bar
        for future in tqdm(as_completed(futures), total=total_to_download, desc="Downloading Images"):
            idx, local_path = future.result()
            results[idx] = local_path
            
    # Add the local image paths back to the dataframe
    df['local_image_path'] = results
    
    # Report stats
    success_count = sum(1 for r in results[:total_to_download] if r is not None)
    print(f"\\nDownload complete: {success_count}/{total_to_download} images successfully downloaded.")
    print(f"Failed to download {total_to_download - success_count} images.")
    
    # Strategy for missing images: We don't remove the product!
    # During OpenCLIP embedding, if 'local_image_path' is None, we will embed the 'product_title' text instead.
    
    # Save the updated catalog
    output_jsonl = input_jsonl.replace(".jsonl", "_with_images.jsonl")
    print(f"Saving updated catalog with local image paths to {output_jsonl}...")
    df.to_json(output_jsonl, orient='records', lines=True, date_format='iso')
    
    return df

if __name__ == "__main__":
    import argparse
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(BASE_DIR, "data", "processed", "cleaned_catalog.jsonl")
    default_output_dir = os.path.join(BASE_DIR, "data", "images")
    
    parser = argparse.ArgumentParser(description="Download Amazon product images")
    parser.add_argument("--input", type=str, default=default_input, help="Path to cleaned JSONL catalog")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Directory to save images (e.g. an external drive path)")
    parser.add_argument("--max-downloads", type=int, default=None, help="Max images to download (use 0 for all)")
    
    args = parser.parse_args()
    
    # Let max_downloads = None if the user passes 0
    limit = None if args.max_downloads == 0 else args.max_downloads
    
    if os.path.exists(args.input):
        print(f"Saving images to: {args.output_dir}")
        process_catalog_images(args.input, args.output_dir, max_downloads=limit)
    else:
        print(f"Error: {args.input} not found. Did you run data_cleaning.py first?")
