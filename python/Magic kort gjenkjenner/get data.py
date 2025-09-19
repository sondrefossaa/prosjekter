import cv2
import numpy as np
import scrython
import requests
import json

import requests
import json
from datetime import datetime

def download_scryfall_data():
    """Download and filter Scryfall bulk data"""
    
    # Set headers as required by Scryfall API
    headers = {
        "User-Agent": "MTGDataProcessor/1.0",
        "Accept": "application/json"
    }
    
    try:
        # Fetch bulk data list
        print("Fetching bulk data list...")
        bulk_response = requests.get("https://api.scryfall.com/bulk-data", headers=headers)
        bulk_response.raise_for_status()  # Raise exception for bad status codes
        bulk_data = bulk_response.json()
        
        # Find the default cards file
        print("Finding default cards file...")
        default_cards_item = next(item for item in bulk_data['data'] if item['type'] == 'default_cards')
        default_cards_uri = default_cards_item['download_uri']
        
        # Download the complete card data
        print("Downloading card data...")
        cards_response = requests.get(default_cards_uri, headers=headers)
        cards_response.raise_for_status()
        cards_list = cards_response.json()
        
        # Filter the data to only include id, prices, and normal image
        print("Filtering data...")
        filtered_data = []
        
        for card in cards_list:
            # Skip digital-only cards and cards without normal images
            if (card.get('digital') or 
                not card.get('image_uris') or 
                'normal' not in card.get('image_uris', {})):
                continue
            
            filtered_card = {
                'id': card.get('id'),
                'name': card.get('name'),  # Including name for readability
                'prices': card.get('prices', {}),
                'image_url': card.get('image_uris', {}).get('small')
            }
            
            # Only add cards that have at least one price or an image
            if (filtered_card['prices'] and any(filtered_card['prices'].values()) or 
                filtered_card['image_url']):
                filtered_data.append(filtered_card)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scryfall_cards_filtered_{timestamp}.json"
        
        # Save filtered data to JSON file
        print(f"Saving data to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"Success! Saved {len(filtered_data)} cards to {filename}")
        return filename
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except StopIteration:
        print("Error: Could not find default cards file in bulk data")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def load_filtered_data(filename):
    """Load the filtered data from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Example usage and data analysis

# Download and save the data
filename = download_scryfall_data()

if filename:
    # Load and analyze the data
    data = load_filtered_data(filename)
    if data:
        print(f"\nData Analysis:")
        print(f"Total cards: {len(data)}")
        
        # Count cards with prices
        cards_with_prices = sum(1 for card in data if card['prices'] and any(card['prices'].values()))
        print(f"Cards with prices: {cards_with_prices}")
        
        # Count cards with images
        cards_with_images = sum(1 for card in data if card['image_url'])
        print(f"Cards with images: {cards_with_images}")
        
        # Show some sample data
        print(f"\nSample cards:")
        for i, card in enumerate(data[:3]):
            print(f"{i+1}. {card['name']}")
            print(f"   ID: {card['id']}")
            print(f"   Prices: {card['prices']}")
            print(f"   Image: {card['image_url']}")
            print()