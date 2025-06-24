# FINAL eBay COLLECTOR FIX
# Replace your src/data_collectors/ebay_collector.py with this

"""
Final Fixed eBay Browse API Collector
"""

import requests
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd
import re
import hashlib

from config.api_keys import EBAY_TOKEN
from config.settings import Settings

class EbayBrowseCollector:
    """Final fixed eBay Browse API collector"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.token = EBAY_TOKEN
        
        # Browse API endpoint
        self.base_url = "https://api.ebay.com/buy/browse/v1"
        
        # Headers for Browse API
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
            'X-EBAY-C-MARKETPLACE-ID': 'EBAY_GB'
        }
        
        self.logger.info("eBay Browse collector initialized")
    
    def search_vehicles(self, brand: str, model: str = None) -> List[Dict]:
        """Search for vehicles using Browse API"""
        
        try:
            # Build search query
            search_query = brand
            if model:
                search_query += f" {model}"
            
            params = {
                'q': search_query,
                'category_ids': '6001',  # eBay Motors > Cars
                'limit': 50,
                'fieldgroups': 'MATCHING_ITEMS,EXTENDED'
            }
            
            url = f"{self.base_url}/item_summary/search"
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            vehicles = self.parse_browse_results(data, brand, model)
            
            self.logger.info(f"Found {len(vehicles)} vehicles for {brand} {model or ''}")
            return vehicles
            
        except Exception as e:
            self.logger.error(f"Error searching for {brand} {model}: {str(e)}")
            return []
    
    def parse_browse_results(self, data: Dict, brand: str, model: str = None) -> List[Dict]:
        """Parse Browse API response with complete data handling"""
        vehicles = []
        
        items = data.get('itemSummaries', [])
        
        for item in items:
            try:
                # Extract basic info
                title = item.get('title', '')
                price_info = item.get('price', {})
                
                # Skip if no price
                if not price_info or not price_info.get('value'):
                    continue
                
                price = float(price_info.get('value', 0))
                
                # Skip unrealistic prices
                if price < 500 or price > 200000:
                    continue
                
                # Create unique item ID
                item_id = item.get('itemId', '')
                if not item_id:
                    item_id = hashlib.md5(f"{title}{price}".encode()).hexdigest()[:10]
                
                # Create base vehicle data with ALL required fields
                vehicle_data = {
                    'item_id': item_id,
                    'title': title,
                    'price': price,
                    'currency': price_info.get('currency', 'GBP'),
                    'condition': item.get('condition', ''),
                    'url': item.get('itemWebUrl', ''),
                    'image_url': item.get('image', {}).get('imageUrl', ''),
                    'seller': item.get('seller', {}).get('username', ''),
                    'location': item.get('itemLocation', {}).get('city', ''),
                    'shipping_cost': self.extract_shipping_cost(item),
                    'collection_timestamp': datetime.now().isoformat(),
                    'source': 'ebay_browse_api',
                    # ENSURE these fields always exist
                    'make': 'Unknown',
                    'model': 'Vehicle',
                    'year': 2018,  # Default year
                    'mileage': None,
                    'fuel_type': None,
                    'transmission': None,
                    'body_type': None,
                    'age': 7  # Default age
                }
                
                # Parse vehicle details from title and override defaults
                parsed_details = self.parse_vehicle_title_enhanced(title, brand, model)
                vehicle_data.update(parsed_details)
                
                # Ensure make and model are not None
                if not vehicle_data.get('make') or vehicle_data['make'] == 'Unknown':
                    vehicle_data['make'] = brand
                
                if not vehicle_data.get('model') or vehicle_data['model'] == 'Vehicle':
                    vehicle_data['model'] = model if model else 'Vehicle'
                
                # Ensure year is valid
                if not vehicle_data.get('year') or vehicle_data['year'] == 2018:
                    vehicle_data['year'] = self.estimate_year_from_price(price)
                
                # Calculate age
                vehicle_data['age'] = datetime.now().year - int(vehicle_data['year'])
                
                vehicles.append(vehicle_data)
                
            except Exception as e:
                self.logger.debug(f"Error parsing item: {str(e)}")
                continue
        
        return vehicles
    
    def extract_shipping_cost(self, item: Dict) -> float:
        """Extract shipping cost from item"""
        try:
            shipping_options = item.get('shippingOptions', [])
            if shipping_options:
                shipping_cost = shipping_options[0].get('shippingCost', {}).get('value', 0)
                return float(shipping_cost) if shipping_cost else 0.0
            return 0.0
        except:
            return 0.0
    
    def parse_vehicle_title_enhanced(self, title: str, brand: str, model: str = None) -> Dict:
        """Enhanced vehicle title parsing with defaults"""
        parsed = {}
        title_upper = title.upper()
        
        # Extract year (4 digits, prioritize recent years)
        year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', title)
        if year_matches:
            # Get the most recent year if multiple found
            years = [int(y) for y in year_matches if 1990 <= int(y) <= datetime.now().year + 1]
            if years:
                parsed['year'] = max(years)
        
        # Extract make (be more flexible)
        make_found = False
        for target_brand in self.settings.TARGET_BRANDS:
            if target_brand.upper() in title_upper:
                parsed['make'] = target_brand
                make_found = True
                break
        
        if not make_found:
            parsed['make'] = brand
        
        # Extract model (enhanced logic)
        model_keywords = [
            'PRIUS', 'CIVIC', 'ACCORD', 'COROLLA', 'CAMRY', 'RAV4', 'CR-V',
            'QASHQAI', 'JUKE', 'X-TRAIL', 'CX-5', 'MAZDA3', 'MAZDA6',
            'OUTBACK', 'FORESTER', 'IMPREZA', 'OUTLANDER', 'ASX',
            'IS', 'GS', 'LS', 'RX', 'NX', 'CT', 'Q50', 'Q30'
        ]
        
        for keyword in model_keywords:
            if keyword in title_upper:
                parsed['model'] = keyword.title()
                break
        
        if not parsed.get('model') and model:
            parsed['model'] = model
        
        # Extract mileage (multiple patterns)
        mileage_patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s*(?:miles?|mi)\b',
            r'(\d+(?:,\d+)?)\s*(?:k|K)\s*(?:miles?|mi)',
            r'(\d+)\s*(?:k|K)\b'
        ]
        
        for pattern in mileage_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                mileage_str = match.group(1).replace(',', '')
                try:
                    if 'k' in match.group(0).lower():
                        parsed['mileage'] = int(mileage_str) * 1000
                    else:
                        parsed['mileage'] = int(mileage_str)
                    break
                except ValueError:
                    continue
        
        # Extract fuel type
        fuel_types = ['PETROL', 'DIESEL', 'HYBRID', 'ELECTRIC', 'PLUG-IN']
        for fuel in fuel_types:
            if fuel in title_upper:
                parsed['fuel_type'] = fuel.title()
                break
        
        # Extract transmission
        if any(word in title_upper for word in ['AUTOMATIC', 'AUTO']):
            parsed['transmission'] = 'Automatic'
        elif any(word in title_upper for word in ['MANUAL', 'MT']):
            parsed['transmission'] = 'Manual'
        
        # Extract body type
        body_types = ['SALOON', 'HATCHBACK', 'ESTATE', 'SUV', 'COUPE', 'CONVERTIBLE']
        for body in body_types:
            if body in title_upper:
                parsed['body_type'] = body.title()
                break
        
        return parsed
    
    def estimate_year_from_price(self, price: float) -> int:
        """Estimate year based on price"""
        current_year = datetime.now().year
        
        if price > 30000:
            return current_year - 2  # Expensive = newer
        elif price > 15000:
            return current_year - 5  # Mid-range
        elif price > 8000:
            return current_year - 8  # Older
        else:
            return current_year - 12  # Budget = older
    
    def collect_uk_vehicle_data(self) -> pd.DataFrame:
        """Main collection method with comprehensive error handling"""
        self.logger.info("Starting UK vehicle data collection with Browse API")
        
        all_vehicles = []
        
        # Strategy: Search by brand
        for brand in self.settings.TARGET_BRANDS:
            try:
                self.logger.info(f"Collecting data for {brand}")
                
                # Brand-only search
                brand_vehicles = self.search_vehicles(brand)
                all_vehicles.extend(brand_vehicles)
                
                # Also search for popular models specifically
                if brand == 'Toyota':
                    model_vehicles = self.search_vehicles(brand, 'Prius')
                    all_vehicles.extend(model_vehicles)
                elif brand == 'Honda':
                    model_vehicles = self.search_vehicles(brand, 'Civic')
                    all_vehicles.extend(model_vehicles)
                elif brand == 'Nissan':
                    model_vehicles = self.search_vehicles(brand, 'Qashqai')
                    all_vehicles.extend(model_vehicles)
                elif brand == 'Mazda':
                    model_vehicles = self.search_vehicles(brand, 'CX-5')
                    all_vehicles.extend(model_vehicles)
                
            except Exception as e:
                self.logger.error(f"Error collecting {brand}: {str(e)}")
                continue
        
        if all_vehicles:
            df = pd.DataFrame(all_vehicles)
            
            # Better deduplication
            original_count = len(df)
            df = df.drop_duplicates(subset=['item_id'], keep='first')
            after_item_id_dedup = len(df)
            
            # Additional deduplication by title + price
            df = df.drop_duplicates(subset=['title', 'price'], keep='first')
            final_count = len(df)
            
            self.logger.info(f"Deduplication: {original_count} -> {after_item_id_dedup} -> {final_count}")
            
            # Clean and validate the data
            df = self.clean_vehicle_data(df)
            
            self.logger.info(f"Collected {len(df)} unique vehicles after cleaning")
            return df
        else:
            return pd.DataFrame()
    
    def clean_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the collected data with guaranteed required columns"""
        
        if df.empty:
            return df
        
        original_count = len(df)
        self.logger.info(f"Starting data cleaning with {original_count} vehicles")
        
        # ENSURE ALL REQUIRED COLUMNS EXIST
        required_columns = {
            'item_id': 'unknown_id',
            'title': 'No Title',
            'make': 'Unknown',
            'model': 'Vehicle',
            'year': 2018,
            'price': 0,
            'currency': 'GBP',
            'mileage': None,
            'fuel_type': None,
            'transmission': None,
            'body_type': None,
            'condition': '',
            'url': '',
            'image_url': '',
            'seller': '',
            'location': '',
            'shipping_cost': 0,
            'source': 'ebay_browse_api',
            'collection_timestamp': datetime.now().isoformat()
        }
        
        # Add missing columns with defaults
        for col, default_value in required_columns.items():
            if col not in df.columns:
                df[col] = default_value
                self.logger.info(f"Added missing column '{col}' with default value")
        
        # Fill missing values
        for col, default_value in required_columns.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_value)
        
        # Clean and validate data types
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2018)
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        df['shipping_cost'] = pd.to_numeric(df['shipping_cost'], errors='coerce').fillna(0)
        
        # Filter valid prices (very lenient)
        df = df[(df['price'] >= 100) & (df['price'] <= 500000)].copy()
        
        # Ensure year is reasonable
        current_year = datetime.now().year
        df.loc[df['year'] > current_year + 1, 'year'] = current_year
        df.loc[df['year'] < 1980, 'year'] = 1990
        
        # Calculate age (NOW SAFE because 'year' column is guaranteed to exist)
        df['age'] = current_year - df['year'].astype(int)
        
        # Ensure essential fields are not null or empty
        df = df[df['item_id'].notna() & (df['item_id'] != '')]
        df = df[df['title'].notna() & (df['title'] != '')]
        df = df[df['price'] > 0]
        
        # Convert object columns to strings to avoid issues
        string_columns = ['make', 'model', 'currency', 'source', 'title']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        self.logger.info(f"Data cleaning complete: {original_count} -> {len(df)} records")
        
        # Log sample if we have data
        if len(df) > 0:
            sample = df.iloc[0]
            self.logger.info(f"Sample cleaned record: {sample['make']} {sample['model']} ({sample['year']}) - £{sample['price']}")
        
        return df