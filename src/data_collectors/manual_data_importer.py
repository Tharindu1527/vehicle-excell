# src/data_collectors/manual_data_importer_fixed.py
"""
FIXED Manual Data Import System - More Tolerant of Different Data Formats
"""

import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import re

from config.settings import Settings

class ManualDataImporter:
    """Fixed manual data importer with better error handling and data tolerance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        
        # Create import directories
        self.import_dir = "data_imports"
        self.uk_import_dir = os.path.join(self.import_dir, "uk_data")
        self.japan_import_dir = os.path.join(self.import_dir, "japan_data")
        self.uploads_dir = os.path.join(self.import_dir, "uploads")
        
        for directory in [self.import_dir, self.uk_import_dir, self.japan_import_dir, self.uploads_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info("Fixed manual data importer initialized")
    
    def import_file(self, file_path: str, data_type: str) -> pd.DataFrame:
        """Import data from CSV or Excel file with extensive error handling"""
        
        self.logger.info(f"Importing {data_type} data from: {file_path}")
        
        try:
            # Read the file
            df = self.read_file_safely(file_path)
            
            if df.empty:
                self.logger.error("File is empty or could not be read")
                return pd.DataFrame()
            
            self.logger.info(f"Read {len(df)} rows from file")
            self.logger.info(f"Columns found: {list(df.columns)}")
            
            # Clean and standardize the data
            if data_type == 'uk':
                df_clean = self.clean_uk_import_data_tolerant(df)
            elif data_type == 'japan':
                df_clean = self.clean_japan_import_data_tolerant(df)
            else:
                self.logger.error(f"Unknown data type: {data_type}")
                return pd.DataFrame()
            
            self.logger.info(f"Cleaned data: {len(df_clean)} valid records")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error importing file: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def read_file_safely(self, file_path: str) -> pd.DataFrame:
        """Safely read CSV or Excel file with multiple fallback options"""
        
        try:
            # Determine file type and read
            if file_path.endswith('.csv'):
                # Try multiple CSV reading strategies
                strategies = [
                    # Standard UTF-8
                    {'encoding': 'utf-8'},
                    # Windows encoding
                    {'encoding': 'latin-1'},
                    {'encoding': 'iso-8859-1'},
                    # Different separators
                    {'encoding': 'utf-8', 'sep': ';'},
                    {'encoding': 'utf-8', 'sep': '\t'},
                    # Handle errors
                    {'encoding': 'utf-8', 'error_bad_lines': False},
                ]
                
                for strategy in strategies:
                    try:
                        self.logger.info(f"Trying CSV read with: {strategy}")
                        df = pd.read_csv(file_path, **strategy)
                        if not df.empty:
                            self.logger.info(f"Successfully read CSV with strategy: {strategy}")
                            return df
                    except Exception as e:
                        self.logger.debug(f"CSV strategy failed: {strategy}, error: {e}")
                        continue
                
                # Last resort: read as text and parse manually
                return self.parse_csv_manually(file_path)
                
            elif file_path.endswith(('.xlsx', '.xls')):
                # Try Excel reading
                try:
                    df = pd.read_excel(file_path)
                    return df
                except Exception as e:
                    self.logger.error(f"Excel read failed: {e}")
                    # Try with different engine
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                        return df
                    except:
                        pass
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"File reading failed completely: {e}")
            return pd.DataFrame()
    
    def parse_csv_manually(self, file_path: str) -> pd.DataFrame:
        """Manually parse CSV as last resort"""
        
        try:
            self.logger.info("Attempting manual CSV parsing...")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if not lines:
                return pd.DataFrame()
            
            # Try to detect delimiter
            first_line = lines[0].strip()
            delimiters = [',', ';', '\t', '|']
            best_delimiter = ','
            max_columns = 0
            
            for delimiter in delimiters:
                columns = len(first_line.split(delimiter))
                if columns > max_columns:
                    max_columns = columns
                    best_delimiter = delimiter
            
            self.logger.info(f"Detected delimiter: '{best_delimiter}', columns: {max_columns}")
            
            # Parse manually
            data = []
            headers = lines[0].strip().split(best_delimiter)
            headers = [h.strip('"').strip() for h in headers]
            
            for line in lines[1:]:
                if line.strip():
                    values = line.strip().split(best_delimiter)
                    values = [v.strip('"').strip() for v in values]
                    # Pad with empty strings if needed
                    while len(values) < len(headers):
                        values.append('')
                    data.append(values[:len(headers)])  # Truncate if too long
            
            df = pd.DataFrame(data, columns=headers)
            self.logger.info(f"Manual parsing successful: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Manual CSV parsing failed: {e}")
            return pd.DataFrame()
    
    def clean_uk_import_data_tolerant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean UK data with maximum tolerance for different formats"""
        
        if df.empty:
            return df
        
        self.logger.info("Cleaning UK import data (tolerant mode)...")
        
        # Create a copy to work with
        df_clean = df.copy()
        
        # Clean column names first
        df_clean.columns = df_clean.columns.str.strip().str.lower()
        
        # Map columns with very flexible matching
        column_mapping = self.get_flexible_uk_column_mapping(df_clean.columns.tolist())
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Log what we found
        self.logger.info(f"Column mapping applied: {column_mapping}")
        self.logger.info(f"Columns after mapping: {list(df_clean.columns)}")
        
        # Check for minimum required data
        has_make = any(col for col in df_clean.columns if 'make' in col.lower() or 'brand' in col.lower())
        has_price = any(col for col in df_clean.columns if 'price' in col.lower() or 'cost' in col.lower() or 'value' in col.lower())
        
        self.logger.info(f"Has make column: {has_make}")
        self.logger.info(f"Has price column: {has_price}")
        
        # If we don't have basic columns, try to extract from any text columns
        if not has_make or not has_price:
            df_clean = self.extract_vehicle_info_from_text(df_clean)
        
        # Ensure essential columns exist with defaults
        required_columns = {
            'item_id': lambda: [f"uk_auto_{i}" for i in range(len(df_clean))],
            'title': lambda: df_clean.iloc[:, 0].astype(str) if len(df_clean.columns) > 0 else ['Unknown Vehicle'] * len(df_clean),
            'make': lambda: self.extract_make_from_available_data(df_clean),
            'model': lambda: ['Vehicle'] * len(df_clean),
            'year': lambda: [2018] * len(df_clean),
            'price': lambda: self.extract_price_from_available_data(df_clean),
            'currency': lambda: ['GBP'] * len(df_clean),
            'mileage': lambda: [None] * len(df_clean),
            'fuel_type': lambda: [None] * len(df_clean),
            'transmission': lambda: [None] * len(df_clean),
            'body_type': lambda: [None] * len(df_clean),
            'condition': lambda: [''] * len(df_clean),
            'url': lambda: [''] * len(df_clean),
            'image_url': lambda: [''] * len(df_clean),
            'seller': lambda: [''] * len(df_clean),
            'location': lambda: [''] * len(df_clean),
            'shipping_cost': lambda: [0] * len(df_clean),
            'source': lambda: ['manual_import_uk'] * len(df_clean),
            'collection_timestamp': lambda: [datetime.now().isoformat()] * len(df_clean)
        }
        
        # Add missing columns
        for col, default_func in required_columns.items():
            if col not in df_clean.columns:
                try:
                    df_clean[col] = default_func()
                except Exception as e:
                    self.logger.warning(f"Error creating default for {col}: {e}")
                    df_clean[col] = [''] * len(df_clean)
        
        # Clean and validate data with tolerance
        df_clean = self.validate_and_clean_uk_data_tolerant(df_clean)
        
        # Calculate age
        current_year = datetime.now().year
        df_clean['age'] = current_year - pd.to_numeric(df_clean['year'], errors='coerce').fillna(2018)
        
        self.logger.info(f"UK data cleaning completed: {len(df_clean)} records")
        return df_clean
    
    def clean_japan_import_data_tolerant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Japan data with maximum tolerance"""
        
        if df.empty:
            return df
        
        self.logger.info("Cleaning Japan import data (tolerant mode)...")
        
        # Create a copy to work with
        df_clean = df.copy()
        
        # Clean column names
        df_clean.columns = df_clean.columns.str.strip().str.lower()
        
        # Map columns flexibly
        column_mapping = self.get_flexible_japan_column_mapping(df_clean.columns.tolist())
        df_clean = df_clean.rename(columns=column_mapping)
        
        self.logger.info(f"Column mapping applied: {column_mapping}")
        
        # Ensure essential columns exist
        required_columns = {
            'auction_id': lambda: [f"jp_auto_{i}" for i in range(len(df_clean))],
            'title': lambda: df_clean.iloc[:, 0].astype(str) if len(df_clean.columns) > 0 else ['Japan Vehicle'] * len(df_clean),
            'make': lambda: self.extract_make_from_available_data(df_clean),
            'model': lambda: ['Vehicle'] * len(df_clean),
            'year': lambda: [2018] * len(df_clean),
            'mileage_km': lambda: [None] * len(df_clean),
            'mileage_miles': lambda: [None] * len(df_clean),
            'final_price_jpy': lambda: self.extract_japan_price_jpy(df_clean),
            'final_price_gbp': lambda: [12000] * len(df_clean),
            'auction_house': lambda: ['Unknown'] * len(df_clean),
            'auction_date': lambda: [datetime.now().isoformat()] * len(df_clean),
            'grade': lambda: ['B'] * len(df_clean),
            'grade_score': lambda: [5.0] * len(df_clean),
            'fuel_type': lambda: ['Petrol'] * len(df_clean),
            'transmission': lambda: ['AT'] * len(df_clean),
            'body_type': lambda: [None] * len(df_clean),
            'colour': lambda: [None] * len(df_clean),
            'estimated_import_cost': lambda: [4000] * len(df_clean),
            'shipping_cost': lambda: [2000] * len(df_clean),
            'import_duty': lambda: [1200] * len(df_clean),
            'vat': lambda: [3000] * len(df_clean),
            'other_costs': lambda: [500] * len(df_clean),
            'total_landed_cost': lambda: [16000] * len(df_clean),
            'source': lambda: ['manual_import_japan'] * len(df_clean),
            'collection_timestamp': lambda: [datetime.now().isoformat()] * len(df_clean)
        }
        
        # Add missing columns
        for col, default_func in required_columns.items():
            if col not in df_clean.columns:
                try:
                    df_clean[col] = default_func()
                except Exception as e:
                    self.logger.warning(f"Error creating default for {col}: {e}")
                    if 'price' in col:
                        df_clean[col] = [12000] * len(df_clean)
                    else:
                        df_clean[col] = [''] * len(df_clean)
        
        # Clean and validate
        df_clean = self.validate_and_clean_japan_data_tolerant(df_clean)
        
        self.logger.info(f"Japan data cleaning completed: {len(df_clean)} records")
        return df_clean
    
    def get_flexible_uk_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Very flexible column mapping for UK data"""
        
        mapping = {}
        
        # Very broad keyword matching
        field_keywords = {
            'make': ['make', 'brand', 'manufacturer', 'marque', 'car_make', 'vehicle_make'],
            'model': ['model', 'car_model', 'vehicle_model', 'variant'],
            'year': ['year', 'reg_year', 'registration_year', 'model_year', 'manufacture_year'],
            'price': ['price', 'cost', 'value', 'amount', 'asking_price', 'sale_price'],
            'mileage': ['mileage', 'miles', 'odometer', 'distance', 'km'],
            'fuel_type': ['fuel', 'fuel_type', 'engine_type', 'power'],
            'transmission': ['transmission', 'gearbox', 'gear'],
            'title': ['title', 'name', 'description', 'vehicle_name', 'ad_title'],
            'location': ['location', 'area', 'region', 'city', 'county']
        }
        
        for standard_field, keywords in field_keywords.items():
            for col in columns:
                col_clean = col.lower().strip()
                for keyword in keywords:
                    if keyword in col_clean:
                        mapping[col] = standard_field
                        break
                if col in mapping:
                    break
        
        return mapping
    
    def get_flexible_japan_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Very flexible column mapping for Japan data"""
        
        mapping = {}
        
        field_keywords = {
            'make': ['make', 'brand', 'manufacturer', 'maker'],
            'model': ['model', 'car_model', 'vehicle_model'],
            'year': ['year', 'model_year', 'manufacture_year'],
            'final_price_jpy': ['price_jpy', 'jpy', 'yen', 'hammer_price', 'auction_price', 'final_price'],
            'final_price_gbp': ['price_gbp', 'gbp', 'pounds', 'sterling'],
            'mileage_km': ['mileage', 'km', 'distance', 'odometer'],
            'auction_house': ['auction', 'house', 'venue', 'location'],
            'grade': ['grade', 'condition', 'rating']
        }
        
        for standard_field, keywords in field_keywords.items():
            for col in columns:
                col_clean = col.lower().strip()
                for keyword in keywords:
                    if keyword in col_clean:
                        mapping[col] = standard_field
                        break
                if col in mapping:
                    break
        
        return mapping
    
    def extract_make_from_available_data(self, df: pd.DataFrame) -> List[str]:
        """Extract vehicle make from any available text data"""
        
        makes = []
        common_makes = ['Toyota', 'Honda', 'Nissan', 'Mazda', 'Subaru', 'Mitsubishi', 'Lexus', 'Infiniti',
                       'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Ford', 'Vauxhall', 'Peugeot']
        
        for _, row in df.iterrows():
            make_found = 'Unknown'
            
            # Check all text columns for make names
            for col in df.columns:
                if df[col].dtype == 'object':  # Text column
                    try:
                        text = str(row[col]).upper()
                        for make in common_makes:
                            if make.upper() in text:
                                make_found = make
                                break
                        if make_found != 'Unknown':
                            break
                    except:
                        continue
            
            makes.append(make_found)
        
        return makes
    
    def extract_price_from_available_data(self, df: pd.DataFrame) -> List[float]:
        """Extract price from any numeric or text data"""
        
        prices = []
        
        for _, row in df.iterrows():
            price_found = 15000.0  # Default price
            
            # Check all columns for numeric values that could be prices
            for col in df.columns:
                try:
                    value = row[col]
                    
                    # Try direct numeric conversion
                    if pd.notna(value):
                        if isinstance(value, (int, float)):
                            if 1000 <= value <= 100000:  # Reasonable price range
                                price_found = float(value)
                                break
                        
                        # Try extracting from text
                        if isinstance(value, str):
                            # Remove currency symbols and extract numbers
                            clean_value = re.sub(r'[£$€,]', '', str(value))
                            numbers = re.findall(r'\d+\.?\d*', clean_value)
                            for num in numbers:
                                try:
                                    num_val = float(num)
                                    if 1000 <= num_val <= 100000:
                                        price_found = num_val
                                        break
                                except:
                                    continue
                            if price_found != 15000.0:
                                break
                except:
                    continue
            
            prices.append(price_found)
        
        return prices
    
    def extract_japan_price_jpy(self, df: pd.DataFrame) -> List[float]:
        """Extract Japanese Yen prices from data"""
        
        prices = []
        
        for _, row in df.iterrows():
            price_found = 2000000.0  # Default JPY price
            
            for col in df.columns:
                try:
                    value = row[col]
                    
                    if pd.notna(value):
                        if isinstance(value, (int, float)):
                            # JPY prices are typically 6-7 digits
                            if 500000 <= value <= 10000000:
                                price_found = float(value)
                                break
                        
                        if isinstance(value, str):
                            clean_value = re.sub(r'[¥,]', '', str(value))
                            numbers = re.findall(r'\d+', clean_value)
                            for num in numbers:
                                try:
                                    num_val = float(num)
                                    if 500000 <= num_val <= 10000000:
                                        price_found = num_val
                                        break
                                except:
                                    continue
                            if price_found != 2000000.0:
                                break
                except:
                    continue
            
            prices.append(price_found)
        
        return prices
    
    def extract_vehicle_info_from_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract vehicle information from any text columns"""
        
        # If we have any text column, try to extract vehicle info
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        if text_columns:
            # Use the first text column as source
            main_text_col = text_columns[0]
            
            # Extract makes
            if 'make' not in df.columns:
                df['make'] = df[main_text_col].apply(self.extract_make_from_text)
            
            # Extract years
            if 'year' not in df.columns:
                df['year'] = df[main_text_col].apply(self.extract_year_from_text)
            
            # Extract prices
            if 'price' not in df.columns:
                df['price'] = df[main_text_col].apply(self.extract_price_from_text)
        
        return df
    
    def extract_make_from_text(self, text: str) -> str:
        """Extract make from a text string"""
        if pd.isna(text):
            return 'Unknown'
        
        text_upper = str(text).upper()
        common_makes = ['TOYOTA', 'HONDA', 'NISSAN', 'MAZDA', 'SUBARU', 'MITSUBISHI', 'LEXUS']
        
        for make in common_makes:
            if make in text_upper:
                return make.title()
        
        return 'Unknown'
    
    def extract_year_from_text(self, text: str) -> int:
        """Extract year from a text string"""
        if pd.isna(text):
            return 2018
        
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', str(text))
        if years:
            return int(max(years))  # Get the latest year
        
        return 2018
    
    def extract_price_from_text(self, text: str) -> float:
        """Extract price from a text string"""
        if pd.isna(text):
            return 15000.0
        
        # Remove currency symbols and extract numbers
        clean_text = re.sub(r'[£$€¥,]', '', str(text))
        numbers = re.findall(r'\d+\.?\d*', clean_text)
        
        for num in numbers:
            try:
                value = float(num)
                if 1000 <= value <= 100000:  # Reasonable price range
                    return value
            except:
                continue
        
        return 15000.0
    
    def validate_and_clean_uk_data_tolerant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate UK data with maximum tolerance"""
        
        original_count = len(df)
        
        # Very lenient price validation
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(15000)
        df = df[df['price'] >= 100]  # Only exclude extremely low prices
        
        # Year validation
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2018)
        current_year = datetime.now().year
        df.loc[df['year'] > current_year + 1, 'year'] = current_year
        df.loc[df['year'] < 1980, 'year'] = 1990
        
        # Clean other fields
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        df['shipping_cost'] = pd.to_numeric(df['shipping_cost'], errors='coerce').fillna(0)
        
        # Ensure text fields are strings
        text_columns = ['make', 'model', 'fuel_type', 'transmission', 'body_type']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        self.logger.info(f"UK validation (tolerant): {original_count} -> {len(df)} records")
        return df
    
    def validate_and_clean_japan_data_tolerant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate Japan data with maximum tolerance"""
        
        original_count = len(df)
        
        # Price validation
        df['final_price_jpy'] = pd.to_numeric(df['final_price_jpy'], errors='coerce').fillna(2000000)
        df['final_price_gbp'] = pd.to_numeric(df['final_price_gbp'], errors='coerce')
        
        # Convert JPY to GBP if missing
        jpy_to_gbp_rate = 0.0055
        mask_missing_gbp = df['final_price_gbp'].isna()
        df.loc[mask_missing_gbp, 'final_price_gbp'] = df.loc[mask_missing_gbp, 'final_price_jpy'] * jpy_to_gbp_rate
        
        # Ensure minimum prices
        df = df[df['final_price_gbp'] >= 100]
        
        # Calculate import costs
        df = self.calculate_import_costs_simple(df)
        
        self.logger.info(f"Japan validation (tolerant): {original_count} -> {len(df)} records")
        return df
    
    def calculate_import_costs_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple import cost calculation"""
        
        for idx, row in df.iterrows():
            price_gbp = row['final_price_gbp']
            
            shipping = price_gbp * 0.15
            duty = price_gbp * 0.10
            vat = (price_gbp + shipping + duty) * 0.20
            other = 500
            
            df.at[idx, 'shipping_cost'] = shipping
            df.at[idx, 'import_duty'] = duty
            df.at[idx, 'vat'] = vat
            df.at[idx, 'other_costs'] = other
            df.at[idx, 'estimated_import_cost'] = shipping + duty + vat + other
            df.at[idx, 'total_landed_cost'] = price_gbp + shipping + duty + vat + other
        
        return df
    
    # Keep all the other methods from the original class for compatibility
    def save_uploaded_file(self, file, data_type: str) -> str:
        """Save uploaded file and return path"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{data_type}_{timestamp}_{file.filename}"
        file_path = os.path.join(self.uploads_dir, safe_filename)
        file.save(file_path)
        self.logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    
    def process_uploaded_file(self, file_path: str, data_type: str) -> Dict:
        """Process uploaded file and return results"""
        try:
            df = self.import_file(file_path, data_type)
            
            if df.empty:
                return {
                    'success': False,
                    'message': 'No valid data found in file - check format and try again',
                    'count': 0
                }
            
            # Move to appropriate directory
            filename = os.path.basename(file_path)
            if data_type == 'uk':
                destination = os.path.join(self.uk_import_dir, filename)
            else:
                destination = os.path.join(self.japan_import_dir, filename)
            
            import shutil
            shutil.copy2(file_path, destination)
            
            return {
                'success': True,
                'message': f'Successfully imported {len(df)} records',
                'count': len(df),
                'file_path': destination
            }
            
        except Exception as e:
            self.logger.error(f"Error processing uploaded file: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing file: {str(e)}',
                'count': 0
            }