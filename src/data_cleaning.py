import pandas as pd
import numpy as np
import os
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

category_keywords = {
    'Laptops': [
        'laptop', 'notebook', 'macbook', 'chromebook', 'ultrabook', 'acer', 'asus', 'dell', 'lenovo', 'hp', 'core',
        'intel', 'ryzen', 'surface', 'thinkpad', 'ideapad'
    ],
    'Phones': [
        'phone', 'iphone', 'smartphone', 'samsung', 'android', 'galaxy', 'pixel', 'oneplus', 'xiaomi', 'oppo',
        'realme', 'huawei', 'vivo', 'nokia', 'motorola'
    ],
    'Headphones': [
        'headphone', 'headset', 'earphone', 'earbuds', 'airpods', 'beats', 'sony wh', 'wireless buds', 'neckband'
    ],
    'Chargers & Cables': [
        'charger', 'charging', 'cable', 'adapter', 'dock', 'usb c', 'type c', 'lightning', 'power adapter', 'usb cable'
    ],
    'Cameras': [
        'camera', 'dslr', 'mirrorless', 'canon', 'nikon', 'gopro', 'instax', 'webcam', 'camcorder', 'security camera'
    ],
    'Storage': [
        'ssd', 'hard drive', 'memory card', 'flash drive', 'pendrive', 'hdd', 'storage', 'micro sd', 'sd card'
    ],
    'Smart Home': [
        'alexa', 'echo', 'smart plug', 'smart bulb', 'smart home', 'nest', 'homekit', 'smart switch'
    ],
    'TV & Display': [
        'monitor', 'display', 'tv', 'screen', 'projector', 'oled', 'led', 'curved monitor', 'uhd', '4k'
    ],
    'Power & Batteries': [
        'battery', 'power bank', 'rechargeable', 'aa', 'aaa', 'portable power', 'cell'
    ],
    'Networking': [
        'wifi', 'router', 'modem', 'ethernet', 'access point', 'mesh', 'network switch'
    ],
    'Wearables': [
        'smartwatch', 'fitness band', 'fitbit', 'watch', 'garmin', 'amazfit'
    ],
    'Speakers': [
        'speaker', 'soundbar', 'subwoofer', 'bluetooth speaker', 'party speaker', 'home theater'
    ],
    'Printers & Scanners': [
        'printer', 'scanner', 'inkjet', 'laserjet', 'photocopier', 'all in one printer'
    ],
    'Gaming': [
        'gaming console', 'playstation', 'ps5', 'ps4', 'xbox', 'nintendo', 'joystick', 'controller', 'gaming mouse',
        'gaming keyboard', 'gaming chair'
    ],
    'Other Electronics': []
}

def assign_category_simple(title):
    title_clean = clean_text(title)
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if kw in title_clean:
                return category
    return 'Other Electronics'

def load_and_clean_data(input_path: str, output_path: str):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Optional: drop completely duplicated rows
    df.drop_duplicates(inplace=True)

    print("Cleaning price_on_variant and current/discounted_price...")
    df['price_on_variant'] = df['price_on_variant'].str.split(":").str.get(1)
    df.loc[~df['price_on_variant'].str.contains(r'\$', na=False), 'price_on_variant'] = np.nan
    df['price_on_variant'] = df['price_on_variant'].str.strip().str.split(" ").str.get(0)
    
    df['current/discounted_price'] = df['current/discounted_price'].fillna(df['price_on_variant'])
    df['current/discounted_price'] = df['current/discounted_price'].str.replace(r"\$", "", regex=True).str.replace(r",", "", regex=True).astype(float)

    print("Cleaning rating and number_of_reviews...")
    df['rating'] = df['rating'].str.replace(r"out of 5 stars", "", regex=True).str.strip().astype(float)
    df['number_of_reviews'] = df['number_of_reviews'].str.replace(",", "").str.strip().astype(float)

    print("Cleaning bought_in_last_month...")
    df['bought_in_last_month'] = df['bought_in_last_month'].str.replace("+ bought in past month","").str.strip().str.replace("K","000")
    df['bought_in_last_month'] = (
        df['bought_in_last_month']
        .where(df['bought_in_last_month'].str.isdigit(), np.nan)
        .astype('Int64')
    )

    print("Cleaning listed_price...")
    df['listed_price'] = df['listed_price'].str.replace("$","").str.replace(",","").str.strip()
    df['listed_price'] = df['listed_price'].replace("No Discount", np.nan)
    df['listed_price'] = df['listed_price'].astype(float)
    df['listed_price'] = df['listed_price'].fillna(df['current/discounted_price'])

    print("Cleaning delivery_details and product_url...")
    df['delivery_details'] = df['delivery_details'].str.extract(r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)?,?\s*(\w+\s+\d{1,2})')
    # Use formatted string to insert the year correctly, ignoring errors explicitly if they arise
    df['delivery_details'] = pd.to_datetime(df['delivery_details'] + ' 2025', errors='coerce')

    amazon_base_url = "https://www.amazon.com"
    df['product_url'] = df['product_url'].apply(
        lambda x: amazon_base_url + x
        if pd.notna(x) and not str(x).startswith(("http://", "https://"))
        else x
    )

    print("Cleaning collected_at and creating category & discount columns...")
    df['collected_at'] = pd.to_datetime(df['collected_at'], errors='coerce')

    df['category'] = df['title'].apply(assign_category_simple)

    df['discount_percentage'] = ((df['listed_price'] - df['current/discounted_price']) / df['listed_price']) * 100
    df['discount_percentage'] = df['discount_percentage'].round(2)

    df.drop(columns=['price_on_variant'], inplace=True)

    print("Renaming columns for consistency...")
    df.rename(columns={
        'title': 'product_title',
        'rating': 'product_rating',
        'number_of_reviews': 'total_reviews',
        'bought_in_last_month': 'purchased_last_month',
        'current/discounted_price': 'discounted_price',
        'listed_price': 'original_price',
        'is_couponed': 'has_coupon',
        'delivery_details': 'delivery_date',
        'sustainability_badges': 'sustainability_tags',
        'image_url': 'product_image_url',
        'product_url': 'product_page_url',
        'collected_at': 'data_collected_at',
        'category': 'product_category',
        'discount_percentage': 'discount_percentage'
    }, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving cleaned dataset to {output_path} as JSONL...")
    # Using orient='records' and lines=True saves the dataframe as JSONL
    df.to_json(output_path, orient='records', lines=True, date_format='iso')
    print("Data cleaning complete!")

if __name__ == "__main__":
    # Adjust paths based on where the raw data is placed
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(BASE_DIR, "data", "raw", "amazon_products_sales_data_uncleaned.csv")
    output_file = os.path.join(BASE_DIR, "data", "processed", "cleaned_catalog.jsonl")
    
    if os.path.exists(input_file):
        load_and_clean_data(input_file, output_file)
    else:
        print(f"Error: Input file {input_file} not found. Please place the Kaggle dataset at this location.")
