"""
UPDATED Web Dashboard Application with Custom UK Column Processing
Flask-based dashboard for vehicle import analysis with specific UK data format
"""

from flask import Flask, render_template, jsonify, request, send_file, redirect, url_for
import pandas as pd
import json
import logging
import traceback
from datetime import datetime, timedelta
import plotly
import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, List
import io
import os
from werkzeug.utils import secure_filename
import re

from src.database.database_manager import DatabaseManager
from src.analysis.scoring_engine import ScoringEngine
from config.settings import Settings

def create_dashboard_app(db_manager: DatabaseManager) -> Flask:
    """Create and configure the Flask dashboard application with UPDATED UK column processing"""
    
    # FIXED: Proper static file paths
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    # Create Flask app with CORRECT configuration
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir,
                static_url_path='/static')
    
    app.secret_key = 'vehicle_import_analyzer_secret_key'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Configure logging
    logger = logging.getLogger(__name__)
    settings = Settings()
    scoring_engine = ScoringEngine()
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def dashboard():
        """Main dashboard page"""
        try:
            logger.info("Dashboard page requested")
            return render_template('dashboard.html')
        except Exception as e:
            logger.error(f"Error loading dashboard template: {str(e)}")
            return f"Error loading dashboard: {str(e)}", 500
    
    # UPDATED: File upload routes for UK and Japan data with specific UK column processing
    @app.route('/api/upload-uk-data', methods=['POST'])
    def upload_uk_data():
        """Upload UK market data from Excel/CSV file with specific column processing"""
        try:
            logger.info("UK data file upload requested")
            
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file uploaded',
                    'data_type': 'UK Market Data'
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected',
                    'data_type': 'UK Market Data'
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid file type. Please upload CSV, XLS, or XLSX files.',
                    'data_type': 'UK Market Data'
                }), 400
            
            # Read the file
            filename = secure_filename(file.filename)
            logger.info(f"Processing UK data file: {filename}")
            
            # Read file based on extension
            try:
                if filename.endswith('.csv'):
                    uk_data = pd.read_csv(file)
                else:  # Excel file
                    uk_data = pd.read_excel(file)
                
                logger.info(f"File read successfully: {len(uk_data)} rows, {len(uk_data.columns)} columns")
                logger.info(f"Available columns: {list(uk_data.columns)}")
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error reading file: {str(e)}',
                    'data_type': 'UK Market Data'
                }), 400
            
            # Process and validate UK data with specific column mapping
            processed_data = process_uk_file_data_custom(uk_data)
            
            if processed_data.empty:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid UK data found in file. Please check the format and ensure required columns are present.',
                    'data_type': 'UK Market Data'
                }), 400
            
            # Store UK data
            db_manager.store_uk_data(processed_data)
            
            logger.info(f"UK data upload completed: {len(processed_data)} vehicles")
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully uploaded {len(processed_data)} UK market vehicles',
                'data_type': 'UK Market Data',
                'count': len(processed_data),
                'source': f'File: {filename}',
                'columns_found': list(uk_data.columns),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error uploading UK data: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'UK data upload failed: {str(e)}',
                'data_type': 'UK Market Data'
            }), 500
    
    def process_uk_file_data_custom(df: pd.DataFrame) -> pd.DataFrame:
        """Process uploaded UK file data with your specific column structure"""
        
        logger.info(f"Processing UK file data with specific columns")
        logger.info(f"Input columns: {list(df.columns)}")
        
        if df.empty:
            return df
        
        # Your specific UK columns:
        # Car Title, Price, Mileage, Registration Date, Image URL, Car Detail URL, 
        # Seller Location, Seller Rating, Features, Price Label
        
        # Create a copy to work with
        df_processed = df.copy()
        
        # Map your specific columns to standard schema
        column_mapping = {}
        
        # Direct mapping for your specific columns
        for col in df_processed.columns:
            col_lower = col.lower().strip()
            
            if 'car title' in col_lower or col_lower == 'title':
                column_mapping[col] = 'title'
            elif col_lower == 'price':
                column_mapping[col] = 'price'
            elif 'mileage' in col_lower:
                column_mapping[col] = 'mileage'
            elif 'registration date' in col_lower or 'reg date' in col_lower:
                column_mapping[col] = 'registration_date'
            elif 'image url' in col_lower or 'image_url' in col_lower:
                column_mapping[col] = 'image_url'
            elif 'car detail url' in col_lower or 'detail url' in col_lower or 'url' in col_lower:
                column_mapping[col] = 'url'
            elif 'seller location' in col_lower or 'location' in col_lower:
                column_mapping[col] = 'location'
            elif 'seller rating' in col_lower or 'rating' in col_lower:
                column_mapping[col] = 'seller_rating'
            elif 'features' in col_lower:
                column_mapping[col] = 'features'
            elif 'price label' in col_lower:
                column_mapping[col] = 'price_label'
        
        logger.info(f"Column mapping: {column_mapping}")
        
        # Rename columns based on mapping
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Extract make, model, year from Car Title
        if 'title' in df_processed.columns:
            df_processed = extract_vehicle_details_from_title(df_processed)
        
        # Process registration date to extract year if year not found in title
        if 'registration_date' in df_processed.columns and 'year' not in df_processed.columns:
            df_processed = extract_year_from_registration(df_processed)
        
        # Clean price data
        if 'price' in df_processed.columns:
            df_processed['price'] = clean_price_column(df_processed['price'])
        
        # Clean mileage data
        if 'mileage' in df_processed.columns:
            df_processed['mileage'] = clean_mileage_column(df_processed['mileage'])
        
        # Extract additional features from Features column
        if 'features' in df_processed.columns:
            df_processed = extract_features_info(df_processed)
        
        # Ensure required columns exist with defaults
        required_columns = {
            'item_id': None,
            'title': 'Vehicle Listing',
            'make': 'Unknown',
            'model': 'Vehicle',
            'year': 2018,
            'price': 0,
            'currency': 'GBP',
            'mileage': None,
            'fuel_type': None,
            'transmission': None,
            'body_type': None,
            'condition': 'Used',
            'url': '',
            'image_url': '',
            'seller': '',
            'location': '',
            'shipping_cost': 0,
            'age': None,
            'source': 'manual_upload_uk',
            'collection_timestamp': datetime.now().isoformat()
        }
        
        for col, default_value in required_columns.items():
            if col not in df_processed.columns:
                df_processed[col] = default_value
        
        # Generate item_id if not present
        if df_processed['item_id'].isna().all() or 'item_id' not in df_processed.columns:
            df_processed['item_id'] = df_processed.apply(
                lambda row: f"uk_upload_{row.name}_{hash(str(row.get('title', 'unknown'))) % 10000}",
                axis=1
            )
        
        # Clean and validate data types
        df_processed['price'] = pd.to_numeric(df_processed['price'], errors='coerce').fillna(0)
        df_processed['year'] = pd.to_numeric(df_processed['year'], errors='coerce').fillna(2018)
        df_processed['mileage'] = pd.to_numeric(df_processed['mileage'], errors='coerce')
        
        # Calculate age
        current_year = datetime.now().year
        df_processed['age'] = current_year - df_processed['year'].astype(int)
        
        # Filter valid records (must have title and reasonable price)
        df_processed = df_processed[
            (df_processed['price'] > 0) & 
            (df_processed['title'].notna()) & 
            (df_processed['title'] != 'Vehicle Listing')
        ]
        
        logger.info(f"Processed UK data: {len(df_processed)} valid records")
        
        # Log sample data
        if len(df_processed) > 0:
            sample = df_processed.iloc[0]
            logger.info(f"Sample processed record: {sample.get('make', 'Unknown')} {sample.get('model', 'Vehicle')} ({sample.get('year', 'Unknown')}) - £{sample.get('price', 0)}")
        
        return df_processed
    
    def extract_vehicle_details_from_title(df: pd.DataFrame) -> pd.DataFrame:
        """Extract make, model, year from Car Title column"""
        
        df = df.copy()
        
        # Initialize columns
        df['make'] = 'Unknown'
        df['model'] = 'Vehicle'
        df['year'] = 2018
        df['fuel_type'] = None
        df['transmission'] = None
        df['body_type'] = None
        
        # Common car makes to look for
        car_makes = [
            'Toyota', 'Honda', 'Nissan', 'Mazda', 'Subaru', 'Mitsubishi', 'Lexus', 'Infiniti',
            'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Ford', 'Vauxhall', 'Peugeot', 'Renault',
            'Hyundai', 'Kia', 'Volvo', 'Jaguar', 'Land Rover', 'Mini', 'Fiat', 'Alfa Romeo',
            'Skoda', 'Seat', 'Citroen', 'Dacia', 'Suzuki', 'Isuzu'
        ]
        
        for idx, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            if not title or title == 'nan':
                continue
            
            title_upper = title.upper()
            
            # Extract year (4 digits, prioritize recent years)
            year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', title)
            if year_matches:
                years = [int(y) for y in year_matches if 1990 <= int(y) <= datetime.now().year + 1]
                if years:
                    df.at[idx, 'year'] = max(years)
            
            # Extract make
            for make in car_makes:
                if make.upper() in title_upper:
                    df.at[idx, 'make'] = make
                    break
            
            # Extract model (after finding make)
            make_found = df.at[idx, 'make']
            if make_found != 'Unknown':
                # Remove make from title and extract model
                title_without_make = title_upper.replace(make_found.upper(), '').strip()
                
                # Common model patterns
                model_patterns = [
                    r'\b(PRIUS|AQUA|COROLLA|CAMRY|RAV4|YARIS|AURIS|AVENSIS)\b',
                    r'\b(CIVIC|ACCORD|CR-V|HR-V|JAZZ|PILOT|INSIGHT)\b',
                    r'\b(QASHQAI|JUKE|X-TRAIL|MICRA|NOTE|LEAF|370Z)\b',
                    r'\b(CX-5|CX-3|MAZDA3|MAZDA6|MX-5|CX-7)\b',
                    r'\b(OUTBACK|FORESTER|IMPREZA|LEGACY|XV)\b',
                    r'\b(OUTLANDER|ASX|L200|SHOGUN|COLT)\b',
                    r'\b(IS|GS|LS|RX|NX|CT|LC)\b',
                    r'\b(Q50|Q30|QX30|QX70|FX)\b'
                ]
                
                for pattern in model_patterns:
                    match = re.search(pattern, title_without_make)
                    if match:
                        df.at[idx, 'model'] = match.group(1).title()
                        break
                
                # If no specific model found, try to extract first word after year
                if df.at[idx, 'model'] == 'Vehicle':
                    words = title_without_make.split()
                    for word in words:
                        if len(word) > 2 and word.isalpha():
                            df.at[idx, 'model'] = word.title()
                            break
            
            # Extract fuel type
            if any(fuel in title_upper for fuel in ['HYBRID', 'HV']):
                df.at[idx, 'fuel_type'] = 'Hybrid'
            elif any(fuel in title_upper for fuel in ['ELECTRIC', 'EV', 'PHEV']):
                df.at[idx, 'fuel_type'] = 'Electric'
            elif 'DIESEL' in title_upper:
                df.at[idx, 'fuel_type'] = 'Diesel'
            elif 'PETROL' in title_upper:
                df.at[idx, 'fuel_type'] = 'Petrol'
            
            # Extract transmission
            if any(trans in title_upper for trans in ['AUTOMATIC', 'AUTO', 'CVT']):
                df.at[idx, 'transmission'] = 'Automatic'
            elif any(trans in title_upper for trans in ['MANUAL', 'MT']):
                df.at[idx, 'transmission'] = 'Manual'
            
            # Extract body type
            if any(body in title_upper for body in ['HATCHBACK', 'HATCH']):
                df.at[idx, 'body_type'] = 'Hatchback'
            elif any(body in title_upper for body in ['SALOON', 'SEDAN']):
                df.at[idx, 'body_type'] = 'Saloon'
            elif any(body in title_upper for body in ['ESTATE', 'TOURING']):
                df.at[idx, 'body_type'] = 'Estate'
            elif any(body in title_upper for body in ['SUV', '4X4']):
                df.at[idx, 'body_type'] = 'SUV'
            elif 'COUPE' in title_upper:
                df.at[idx, 'body_type'] = 'Coupe'
        
        return df
    
    def extract_year_from_registration(df: pd.DataFrame) -> pd.DataFrame:
        """Extract year from registration date if year not found in title"""
        
        df = df.copy()
        
        for idx, row in df.iterrows():
            if df.at[idx, 'year'] == 2018:  # Default year, try to get from registration
                reg_date = str(row.get('registration_date', ''))
                
                # Try different date formats
                year_from_reg = None
                try:
                    # Try common date formats
                    for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y', '%d-%m-%Y']:
                        try:
                            parsed_date = datetime.strptime(reg_date, date_format)
                            year_from_reg = parsed_date.year
                            break
                        except ValueError:
                            continue
                    
                    # If still no year, try to extract 4-digit year from string
                    if not year_from_reg:
                        year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', reg_date)
                        if year_matches:
                            year_from_reg = int(year_matches[0])
                    
                    if year_from_reg and 1990 <= year_from_reg <= datetime.now().year:
                        df.at[idx, 'year'] = year_from_reg
                        
                except Exception as e:
                    continue
        
        return df
    
    def clean_price_column(price_series: pd.Series) -> pd.Series:
        """Clean price column to extract numeric values"""
        
        def clean_price_value(price_val):
            if pd.isna(price_val):
                return 0
            
            price_str = str(price_val).strip()
            
            # Remove currency symbols and common text
            price_str = re.sub(r'[£$€¥,]', '', price_str)
            price_str = re.sub(r'[^\d.]', '', price_str)
            
            try:
                return float(price_str)
            except (ValueError, TypeError):
                return 0
        
        return price_series.apply(clean_price_value)
    
    def clean_mileage_column(mileage_series: pd.Series) -> pd.Series:
        """Clean mileage column to extract numeric values"""
        
        def clean_mileage_value(mileage_val):
            if pd.isna(mileage_val):
                return None
            
            mileage_str = str(mileage_val).strip()
            
            # Remove common text and extract number
            mileage_str = re.sub(r'[^\d,.]', '', mileage_str)
            mileage_str = mileage_str.replace(',', '')
            
            try:
                return float(mileage_str)
            except (ValueError, TypeError):
                return None
        
        return mileage_series.apply(clean_mileage_value)
    
    def extract_features_info(df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional information from Features column"""
        
        df = df.copy()
        
        for idx, row in df.iterrows():
            features = str(row.get('features', '')).upper()
            
            # Update fuel type if not already set
            if not df.at[idx, 'fuel_type']:
                if 'HYBRID' in features:
                    df.at[idx, 'fuel_type'] = 'Hybrid'
                elif 'ELECTRIC' in features:
                    df.at[idx, 'fuel_type'] = 'Electric'
                elif 'DIESEL' in features:
                    df.at[idx, 'fuel_type'] = 'Diesel'
                elif 'PETROL' in features:
                    df.at[idx, 'fuel_type'] = 'Petrol'
            
            # Update transmission if not already set
            if not df.at[idx, 'transmission']:
                if any(trans in features for trans in ['AUTOMATIC', 'AUTO', 'CVT']):
                    df.at[idx, 'transmission'] = 'Automatic'
                elif 'MANUAL' in features:
                    df.at[idx, 'transmission'] = 'Manual'
        
        return df
    
    @app.route('/api/upload-japan-data', methods=['POST'])
    def upload_japan_data():
        """Upload Japan auction data from Excel/CSV file"""
        try:
            logger.info("Japan data file upload requested")
            
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file uploaded',
                    'data_type': 'Japan Auction Data'
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected',
                    'data_type': 'Japan Auction Data'
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid file type. Please upload CSV, XLS, or XLSX files.',
                    'data_type': 'Japan Auction Data'
                }), 400
            
            # Read the file
            filename = secure_filename(file.filename)
            logger.info(f"Processing Japan data file: {filename}")
            
            # Read file based on extension
            try:
                if filename.endswith('.csv'):
                    japan_data = pd.read_csv(file)
                else:  # Excel file
                    japan_data = pd.read_excel(file)
                
                logger.info(f"File read successfully: {len(japan_data)} rows, {len(japan_data.columns)} columns")
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error reading file: {str(e)}',
                    'data_type': 'Japan Auction Data'
                }), 400
            
            # Process and validate Japan data (keeping original function)
            processed_data = process_japan_file_data(japan_data)
            
            if processed_data.empty:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid Japan data found in file. Please check the format.',
                    'data_type': 'Japan Auction Data'
                }), 400
            
            # Store Japan data
            db_manager.store_japan_data(processed_data)
            
            logger.info(f"Japan data upload completed: {len(processed_data)} vehicles")
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully uploaded {len(processed_data)} Japan auction vehicles',
                'data_type': 'Japan Auction Data',
                'count': len(processed_data),
                'source': f'File: {filename}',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error uploading Japan data: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Japan data upload failed: {str(e)}',
                'data_type': 'Japan Auction Data'
            }), 500
    
    def process_japan_file_data(df: pd.DataFrame) -> pd.DataFrame:
        """Process uploaded Japan file data and map to required schema (keeping original)"""
        
        logger.info(f"Processing Japan file data with columns: {list(df.columns)}")
        
        # Create mapping for common column names (case-insensitive)
        column_mapping = {}
        df_columns_lower = [col.lower().strip() for col in df.columns]
        
        # Map common variations to standard column names
        mappings = {
            'make': ['make', 'manufacturer', 'brand'],
            'model': ['model', 'vehicle_model', 'car_model'],
            'year': ['year', 'manufacture_year', 'model_year'],
            'final_price_jpy': ['final_price_jpy', 'price_jpy', 'auction_price_jpy', 'winning_bid_jpy'],
            'final_price_gbp': ['final_price_gbp', 'price_gbp', 'auction_price_gbp', 'price_pound'],
            'mileage_km': ['mileage_km', 'mileage', 'odometer_km', 'distance_km'],
            'grade': ['grade', 'auction_grade', 'condition_grade'],
            'fuel_type': ['fuel_type', 'fuel', 'engine_type'],
            'transmission': ['transmission', 'gearbox', 'gear'],
            'body_type': ['body_type', 'body_style', 'type'],
            'colour': ['colour', 'color', 'paint'],
            'auction_house': ['auction_house', 'auction_location', 'seller'],
            'auction_date': ['auction_date', 'sale_date', 'date'],
            'title': ['title', 'description', 'name', 'vehicle_name']
        }
        
        for standard_col, variations in mappings.items():
            for variation in variations:
                if variation in df_columns_lower:
                    original_col = df.columns[df_columns_lower.index(variation)]
                    column_mapping[original_col] = standard_col
                    break
        
        logger.info(f"Column mapping: {column_mapping}")
        
        # Rename columns based on mapping
        df_processed = df.rename(columns=column_mapping)
        
        # Ensure required columns exist with defaults
        required_columns = {
            'auction_id': None,
            'title': 'Japan Auction Vehicle',
            'make': 'Toyota',
            'model': 'Vehicle',
            'year': 2018,
            'mileage_km': None,
            'mileage_miles': None,
            'final_price_jpy': 0,
            'final_price_gbp': 0,
            'auction_house': 'Japanese Auction',
            'auction_date': datetime.now().strftime('%Y-%m-%d'),
            'grade': 'B',
            'grade_score': 6.0,
            'fuel_type': 'Petrol',
            'transmission': 'AT',
            'body_type': 'Sedan',
            'colour': 'White',
            'estimated_import_cost': 0,
            'shipping_cost': 0,
            'import_duty': 0,
            'vat': 0,
            'other_costs': 0,
            'total_landed_cost': 0,
            'source': 'manual_upload_japan',
            'collection_timestamp': datetime.now().isoformat()
        }
        
        for col, default_value in required_columns.items():
            if col not in df_processed.columns:
                df_processed[col] = default_value
        
        # Generate auction_id if not present
        if df_processed['auction_id'].isna().all():
            df_processed['auction_id'] = df_processed.apply(
                lambda row: f"jp_upload_{row.name}_{hash(str(row.get('make', 'toyota')) + str(row.get('model', 'vehicle'))) % 10000}",
                axis=1
            )
        
        # Clean and validate data
        df_processed['final_price_jpy'] = pd.to_numeric(df_processed['final_price_jpy'], errors='coerce').fillna(0)
        df_processed['final_price_gbp'] = pd.to_numeric(df_processed['final_price_gbp'], errors='coerce').fillna(0)
        df_processed['year'] = pd.to_numeric(df_processed['year'], errors='coerce').fillna(2018)
        df_processed['mileage_km'] = pd.to_numeric(df_processed['mileage_km'], errors='coerce')
        
        # Convert JPY to GBP if GBP not provided
        if df_processed['final_price_gbp'].sum() == 0 and df_processed['final_price_jpy'].sum() > 0:
            df_processed['final_price_gbp'] = df_processed['final_price_jpy'] * 0.0055  # Approximate conversion
        
        # Calculate mileage in miles if not provided
        if df_processed['mileage_miles'].isna().all() and not df_processed['mileage_km'].isna().all():
            df_processed['mileage_miles'] = df_processed['mileage_km'] * 0.621371
        
        # Calculate import costs if not provided
        mask_no_costs = df_processed['total_landed_cost'] == 0
        if mask_no_costs.any():
            estimated_import_cost = df_processed.loc[mask_no_costs, 'final_price_gbp'] * 0.35
            df_processed.loc[mask_no_costs, 'estimated_import_cost'] = estimated_import_cost
            df_processed.loc[mask_no_costs, 'shipping_cost'] = estimated_import_cost * 0.3
            df_processed.loc[mask_no_costs, 'import_duty'] = estimated_import_cost * 0.3
            df_processed.loc[mask_no_costs, 'vat'] = estimated_import_cost * 0.4
            df_processed.loc[mask_no_costs, 'total_landed_cost'] = df_processed.loc[mask_no_costs, 'final_price_gbp'] + estimated_import_cost
        
        # Convert grade to score
        grade_scores = {'S': 9, 'A': 7, 'B': 6, 'C': 4, 'D': 2}
        df_processed['grade_score'] = df_processed['grade'].map(grade_scores).fillna(6.0)
        
        # Filter valid records
        df_processed = df_processed[
            (df_processed['final_price_gbp'] > 0) & 
            (df_processed['make'].notna())
        ]
        
        logger.info(f"Processed Japan data: {len(df_processed)} valid records")
        
        return df_processed
    
    @app.route('/api/download-template/<data_type>')
    def download_template(data_type):
        """Download CSV template for UK or Japan data with updated UK format"""
        try:
            if data_type == 'uk':
                # Create UK template with your specific columns
                template_data = {
                    'Car Title': [
                        '2019 Toyota Prius 1.8 VVT-i Hybrid Excel CVT Euro 6 5dr',
                        '2018 Honda Civic 1.5 VTEC Turbo SR 5dr CVT',
                        '2017 Nissan Qashqai 1.2 DIG-T Acenta Premium 5dr'
                    ],
                    'Price': [18500, 16200, 15800],
                    'Mileage': [45000, 38000, 42000],
                    'Registration Date': ['2019-03-15', '2018-09-20', '2017-06-10'],
                    'Image URL': [
                        'https://example.com/image1.jpg',
                        'https://example.com/image2.jpg', 
                        'https://example.com/image3.jpg'
                    ],
                    'Car Detail URL': [
                        'https://example.com/car/12345',
                        'https://example.com/car/12346',
                        'https://example.com/car/12347'
                    ],
                    'Seller Location': ['London', 'Manchester', 'Birmingham'],
                    'Seller Rating': [4.8, 4.5, 4.2],
                    'Features': [
                        'Hybrid, Automatic, Air Con, Sat Nav',
                        'Petrol, Manual, Air Con, Bluetooth',
                        'Petrol, CVT, Air Con, Parking Sensors'
                    ],
                    'Price Label': ['£18,500', '£16,200', '£15,800']
                }
                filename = 'uk_data_template.csv'
                
            elif data_type == 'japan':
                # Create Japan template (keeping original)
                template_data = {
                    'make': ['Toyota', 'Honda', 'Nissan'],
                    'model': ['Prius', 'Civic', 'Qashqai'],
                    'year': [2019, 2018, 2017],
                    'final_price_jpy': [2180000, 1909000, 1782000],
                    'final_price_gbp': [12000, 10500, 9800],
                    'mileage_km': [60000, 55000, 58000],
                    'grade': ['B', 'A', 'B'],
                    'fuel_type': ['Hybrid', 'Petrol', 'Petrol'],
                    'transmission': ['CVT', 'Manual', 'CVT'],
                    'body_type': ['Hatchback', 'Hatchback', 'SUV'],
                    'colour': ['White', 'Red', 'Black'],
                    'auction_house': ['USS Tokyo', 'TAA Kansai', 'USS Osaka'],
                    'auction_date': ['2025-06-15', '2025-06-14', '2025-06-13'],
                    'title': ['2019 Toyota Prius Hybrid', '2018 Honda Civic Type R', '2017 Nissan Qashqai']
                }
                filename = 'japan_data_template.csv'
                
            else:
                return jsonify({'error': 'Invalid template type'}), 400
            
            # Create DataFrame and CSV
            df = pd.DataFrame(template_data)
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            # Convert to bytes
            csv_data = output.getvalue().encode('utf-8')
            
            return send_file(
                io.BytesIO(csv_data),
                as_attachment=True,
                download_name=filename,
                mimetype='text/csv'
            )
            
        except Exception as e:
            logger.error(f"Error creating template: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # Keep all the existing routes from the original dashboard
    @app.route('/api/run-analysis', methods=['POST'])
    def run_analysis():
        """Run profitability analysis on collected data"""
        try:
            logger.info("Analysis run requested")
            
            # Get latest data from database
            uk_data = db_manager.get_uk_data(days_back=30)
            japan_data = db_manager.get_japan_data(days_back=30)
            
            if uk_data.empty:
                return jsonify({
                    'status': 'error',
                    'message': 'No UK data available. Please upload UK data first.',
                    'data_type': 'Analysis'
                }), 400
            
            if japan_data.empty:
                return jsonify({
                    'status': 'error',
                    'message': 'No Japan data available. Please upload Japan data first.',
                    'data_type': 'Analysis'
                }), 400
            
            # Run analysis
            from src.analysis.profitability_analyzer import ProfitabilityAnalyzer
            profitability_analyzer = ProfitabilityAnalyzer()
            
            logger.info("Running profitability analysis...")
            profitability_results = profitability_analyzer.analyze(uk_data, japan_data)
            
            if profitability_results.empty:
                return jsonify({
                    'status': 'warning',
                    'message': 'No profitable opportunities found',
                    'data_type': 'Analysis',
                    'count': 0
                })
            
            # Generate scores
            logger.info("Generating investment scores...")
            scores = scoring_engine.calculate_scores(profitability_results)
            
            # Store analysis results
            db_manager.store_analysis_results(profitability_results, scores)
            
            logger.info(f"Analysis completed: {len(scores)} opportunities")
            
            return jsonify({
                'status': 'success',
                'message': f'Analysis completed: {len(scores)} profitable opportunities found',
                'data_type': 'Analysis Results',
                'count': len(scores),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Analysis failed: {str(e)}',
                'data_type': 'Analysis'
            }), 500
    
    @app.route('/api/data-status')
    def data_status():
        """Get current data status for all sources"""
        try:
            logger.info("Data status requested")
            
            # Get data counts
            uk_data = db_manager.get_uk_data(days_back=7)
            japan_data = db_manager.get_japan_data(days_back=7)
            analysis_results = db_manager.get_latest_analysis_results(limit=100)
            
            uk_count = len(uk_data)
            japan_count = len(japan_data)
            analysis_count = len(analysis_results)
            
            # Get last collection times
            uk_last_collection = uk_data['collection_timestamp'].max() if not uk_data.empty else None
            japan_last_collection = japan_data['collection_timestamp'].max() if not japan_data.empty else None
            analysis_last_run = analysis_results['analysis_timestamp'].max() if not analysis_results.empty else None
            
            return jsonify({
                'uk_data': {
                    'count': uk_count,
                    'last_collection': uk_last_collection,
                    'status': 'available' if uk_count > 0 else 'empty'
                },
                'japan_data': {
                    'count': japan_count,
                    'last_collection': japan_last_collection,
                    'status': 'available' if japan_count > 0 else 'empty'
                },
                'analysis_data': {
                    'count': analysis_count,
                    'last_analysis': analysis_last_run,
                    'status': 'available' if analysis_count > 0 else 'empty'
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting data status: {str(e)}")
            return jsonify({
                'error': str(e)
            }), 500
    
    @app.route('/api/test')
    def api_test():
        """Test endpoint to verify everything is working"""
        try:
            logger.info("API test endpoint called")
            
            # Test database connection
            stats = db_manager.get_database_stats()
            logger.info(f"Database stats retrieved: {stats}")
            
            # Test data retrieval
            uk_data = db_manager.get_uk_data(days_back=30)
            japan_data = db_manager.get_japan_data(days_back=30)
            analysis_results = db_manager.get_latest_analysis_results(limit=5)
            
            uk_count = len(uk_data) if not uk_data.empty else 0
            japan_count = len(japan_data) if not japan_data.empty else 0
            analysis_count = len(analysis_results) if not analysis_results.empty else 0
            
            logger.info(f"Data counts - UK: {uk_count}, Japan: {japan_count}, Analysis: {analysis_count}")
            
            return jsonify({
                'status': 'ok',
                'database_stats': stats,
                'data_counts': {
                    'uk_records': uk_count,
                    'japan_records': japan_count,
                    'analysis_records': analysis_count
                },
                'database_path': db_manager.db_path,
                'database_exists': os.path.exists(db_manager.db_path),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in test endpoint: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500
    
    @app.route('/api/summary')
    def api_summary():
        """Get dashboard summary data"""
        try:
            logger.info("API summary endpoint called")
            
            if not os.path.exists(db_manager.db_path):
                logger.error(f"Database file does not exist: {db_manager.db_path}")
                return jsonify({
                    'error': 'Database not found',
                    'message': 'Run system_rebuild.py to initialize the system'
                }), 500
            
            # Get market summary
            try:
                market_summary = db_manager.get_market_summary()
                logger.info(f"Market summary retrieved: {bool(market_summary)}")
            except Exception as e:
                logger.error(f"Error getting market summary: {e}")
                market_summary = {}
            
            # Get latest analysis results
            try:
                analysis_results = db_manager.get_latest_analysis_results(limit=10)
                logger.info(f"Analysis results count: {len(analysis_results)}")
                
                top_opportunities = []
                if not analysis_results.empty:
                    top_5 = analysis_results.head(5)
                    for _, row in top_5.iterrows():
                        try:
                            opportunity = {
                                'make': str(row.get('make', 'Unknown')),
                                'model': str(row.get('model', 'Vehicle')),
                                'year': int(row.get('year', 2018)),
                                'final_score': float(row.get('final_score', 0)),
                                'profit_margin': float(row.get('profit_margin', 0)),
                                'gross_profit': float(row.get('gross_profit', 0)),
                                'roi': float(row.get('roi', 0)),
                                'investment_category': str(row.get('investment_category', 'Unknown'))
                            }
                            top_opportunities.append(opportunity)
                        except Exception as e:
                            logger.error(f"Error processing opportunity: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error getting analysis results: {e}")
                analysis_results = pd.DataFrame()
                top_opportunities = []
            
            # Get database stats
            try:
                db_stats = db_manager.get_database_stats()
                logger.info(f"Database stats: {db_stats}")
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                db_stats = {}
            
            summary = {
                'market_summary': market_summary,
                'top_opportunities': top_opportunities,
                'database_stats': db_stats,
                'has_data': len(top_opportunities) > 0,
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"Summary prepared with {len(top_opportunities)} opportunities")
            return jsonify(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Please check logs and ensure database is properly initialized'
            }), 500
    
    @app.route('/api/top-vehicles')
    def api_top_vehicles():
        """Get top vehicle opportunities"""
        try:
            limit = request.args.get('limit', 20, type=int)
            logger.info(f"Top vehicles endpoint called with limit: {limit}")
            
            analysis_results = db_manager.get_latest_analysis_results(limit=limit)
            
            if analysis_results.empty:
                logger.warning("No analysis results available")
                return jsonify({
                    'top_vehicles': [],
                    'total_analyzed': 0,
                    'message': 'No analysis results available. Upload data and run analysis first.'
                })
            
            # Generate recommendations safely
            try:
                recommendations = scoring_engine.generate_top_recommendations(analysis_results, top_n=limit)
                logger.info(f"Generated recommendations for {len(analysis_results)} vehicles")
                return jsonify(recommendations)
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                # Fallback: return basic vehicle list
                vehicles = []
                for i, (_, row) in enumerate(analysis_results.head(limit).iterrows(), 1):
                    try:
                        vehicle = {
                            'rank': i,
                            'make': str(row.get('make', 'Unknown')),
                            'model': str(row.get('model', 'Vehicle')),
                            'year': int(row.get('year', 2018)),
                            'final_score': float(row.get('final_score', 0)),
                            'profit_margin': float(row.get('profit_margin', 0)),
                            'expected_profit': float(row.get('gross_profit', 0)),
                            'roi': float(row.get('roi', 0)),
                            'risk_level': 'Medium',
                            'risk_score': float(row.get('risk_score', 5)),
                            'investment_category': str(row.get('investment_category', 'Unknown'))
                        }
                        vehicles.append(vehicle)
                    except Exception as ve:
                        logger.error(f"Error processing vehicle {i}: {ve}")
                        continue
                
                return jsonify({
                    'top_vehicles': vehicles,
                    'total_analyzed': len(analysis_results),
                    'message': 'Basic vehicle data (scoring engine error)'
                })
            
        except Exception as e:
            logger.error(f"Error getting top vehicles: {str(e)}")
            return jsonify({
                'error': str(e),
                'top_vehicles': [],
                'message': 'Error retrieving vehicle data'
            }), 500
    
    @app.route('/api/market-trends')
    def api_market_trends():
        """Get market trend data for charts"""
        try:
            logger.info("Market trends endpoint called")
            
            # Get recent data for trends
            uk_data = db_manager.get_uk_data(days_back=30)
            japan_data = db_manager.get_japan_data(days_back=30)
            
            trends = generate_trend_charts(uk_data, japan_data)
            
            # If no real data, provide sample trend data
            if not trends:
                trends = {
                    'uk_price_by_make': {
                        'x': ['Toyota', 'Honda', 'Nissan', 'Mazda'],
                        'y': [18500, 16200, 15800, 14500],
                        'type': 'bar',
                        'name': 'Average UK Price by Make'
                    },
                    'japan_price_by_make': {
                        'x': ['Toyota', 'Honda', 'Nissan', 'Mazda'],
                        'y': [12000, 10500, 9800, 8500],
                        'type': 'bar',
                        'name': 'Average Japan Auction Price by Make'
                    }
                }
            
            return jsonify(trends)
            
        except Exception as e:
            logger.error(f"Error generating market trends: {str(e)}")
            return jsonify({
                'uk_price_by_make': {'x': [], 'y': []},
                'japan_price_by_make': {'x': [], 'y': []},
                'message': 'No trend data available'
            })
    
    @app.route('/api/export/excel')
    def api_export_excel():
        """Export analysis results to Excel"""
        try:
            analysis_results = db_manager.get_latest_analysis_results(limit=1000)
            
            if analysis_results.empty:
                return jsonify({'error': 'No data to export'}), 400
            
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                analysis_results.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                summary_data = generate_export_summary(analysis_results)
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                top_20 = analysis_results.head(20)
                top_20.to_excel(writer, sheet_name='Top 20 Opportunities', index=False)
            
            output.seek(0)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'vehicle_analysis_{timestamp}.xlsx'
            
            return send_file(
                output,
                as_attachment=True,
                download_name=filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/portfolio-optimizer')
    def api_portfolio_optimizer():
        """Get portfolio optimization suggestions"""
        try:
            budget = request.args.get('budget', 100000, type=float)
            
            analysis_results = db_manager.get_latest_analysis_results(limit=200)
            
            if analysis_results.empty:
                return jsonify({
                    'error': 'No analysis data available',
                    'message': 'Upload data and run analysis first'
                })
            
            portfolio = scoring_engine.calculate_portfolio_optimization(analysis_results, budget)
            
            return jsonify(portfolio)
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/search')
    def api_search():
        """Search vehicles by make, model, or other criteria"""
        try:
            query = request.args.get('q', '').strip()
            limit = request.args.get('limit', 50, type=int)
            
            if not query:
                return jsonify({'error': 'Search query is required'}), 400
            
            analysis_results = db_manager.get_latest_analysis_results(limit=1000)
            
            if analysis_results.empty:
                return jsonify({'results': [], 'message': 'No data to search'})
            
            filtered_results = analysis_results[
                analysis_results['make'].str.contains(query, case=False, na=False) |
                analysis_results['model'].str.contains(query, case=False, na=False)
            ].head(limit)
            
            results = []
            for _, row in filtered_results.iterrows():
                try:
                    result = {
                        'make': str(row.get('make', 'Unknown')),
                        'model': str(row.get('model', 'Vehicle')),
                        'year': int(row.get('year', 2018)),
                        'final_score': float(row.get('final_score', 0)),
                        'profit_margin': float(row.get('profit_margin', 0)),
                        'expected_profit': float(row.get('gross_profit', 0))
                    }
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue
            
            return jsonify({
                'results': results,
                'total_found': len(results),
                'query': query
            })
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/status')
    def status_page():
        """Simple status page to debug issues"""
        try:
            stats = db_manager.get_database_stats()
            uk_count = len(db_manager.get_uk_data(days_back=7))
            japan_count = len(db_manager.get_japan_data(days_back=7))
            analysis_count = len(db_manager.get_latest_analysis_results(limit=10))
            
            status_html = f"""
            <html>
            <head><title>System Status</title></head>
            <body>
                <h1>Vehicle Import Analyzer - System Status</h1>
                <h2>Database Status</h2>
                <p>Database file: {db_manager.db_path}</p>
                <p>Database exists: {os.path.exists(db_manager.db_path)}</p>
                <p>Database stats: {stats}</p>
                
                <h2>Data Counts (Last 7 days)</h2>
                <p>UK Records: {uk_count}</p>
                <p>Japan Records: {japan_count}</p>
                <p>Analysis Results: {analysis_count}</p>
                
                <h2>Test Links</h2>
                <p><a href="/api/test">Test API</a></p>
                <p><a href="/api/summary">Summary API</a></p>
                <p><a href="/api/data-status">Data Status API</a></p>
                <p><a href="/">Main Dashboard</a></p>
                
                <h2>File Upload APIs</h2>
                <p>POST <a href="/api/upload-uk-data">/api/upload-uk-data</a> - Upload UK market data (CSV/Excel)</p>
                <p>POST <a href="/api/upload-japan-data">/api/upload-japan-data</a> - Upload Japan auction data (CSV/Excel)</p>
                <p>POST <a href="/api/run-analysis">/api/run-analysis</a> - Run profitability analysis</p>
                
                <h2>Download Templates</h2>
                <p><a href="/api/download-template/uk">Download UK Data Template (CSV)</a> - Updated format</p>
                <p><a href="/api/download-template/japan">Download Japan Data Template (CSV)</a></p>
                
                <h2>UK Data Format</h2>
                <p>Expected columns: Car Title, Price, Mileage, Registration Date, Image URL, Car Detail URL, Seller Location, Seller Rating, Features, Price Label</p>
                
                <h2>Actions</h2>
                <p>If no data: Upload CSV/Excel files using the dashboard</p>
                
                <p>Generated at: {datetime.now().isoformat()}</p>
            </body>
            </html>
            """
            return status_html
            
        except Exception as e:
            return f"<h1>Error</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>"
    
    def generate_trend_charts(uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> Dict:
        """Generate chart data for market trends"""
        
        charts = {}
        
        try:
            if not uk_data.empty and 'make' in uk_data.columns and 'price' in uk_data.columns:
                make_prices = uk_data.groupby('make')['price'].agg(['mean', 'count']).reset_index()
                make_prices = make_prices[make_prices['count'] >= 2]
                
                if not make_prices.empty:
                    charts['uk_price_by_make'] = {
                        'x': make_prices['make'].tolist(),
                        'y': make_prices['mean'].round(0).tolist(),
                        'type': 'bar',
                        'name': 'Average UK Price by Make'
                    }
            
            if not japan_data.empty and 'make' in japan_data.columns and 'final_price_gbp' in japan_data.columns:
                japan_make_prices = japan_data.groupby('make')['final_price_gbp'].agg(['mean', 'count']).reset_index()
                japan_make_prices = japan_make_prices[japan_make_prices['count'] >= 2]
                
                if not japan_make_prices.empty:
                    charts['japan_price_by_make'] = {
                        'x': japan_make_prices['make'].tolist(),
                        'y': japan_make_prices['mean'].round(0).tolist(),
                        'type': 'bar',
                        'name': 'Average Japan Auction Price by Make'
                    }
            
        except Exception as e:
            logger.error(f"Error generating trend charts: {str(e)}")
        
        return charts
    
    def generate_export_summary(analysis_results: pd.DataFrame) -> Dict:
        """Generate summary data for Excel export"""
        
        try:
            if analysis_results.empty:
                return {'error': 'No data to summarize'}
            
            return {
                'Total Vehicles Analyzed': len(analysis_results),
                'Average Profit Margin': f"{analysis_results['profit_margin'].mean():.1%}" if 'profit_margin' in analysis_results.columns else 'N/A',
                'Average ROI': f"{analysis_results['roi'].mean():.1%}" if 'roi' in analysis_results.columns else 'N/A',
                'Best Opportunity': f"{analysis_results.iloc[0]['make']} {analysis_results.iloc[0]['model']}" if len(analysis_results) > 0 else 'N/A',
                'Highest Profit Margin': f"{analysis_results['profit_margin'].max():.1%}" if 'profit_margin' in analysis_results.columns else 'N/A',
                'Total Profit Potential': f"£{analysis_results['gross_profit'].sum():,.0f}" if 'gross_profit' in analysis_results.columns else 'N/A',
                'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Analysis Period': '30 days',
                'Top Make by Count': analysis_results['make'].value_counts().index[0] if not analysis_results.empty else 'N/A',
                'Average Final Score': f"{analysis_results['final_score'].mean():.1f}" if 'final_score' in analysis_results.columns else 'N/A'
            }
        except Exception as e:
            logger.error(f"Error generating export summary: {str(e)}")
            return {'error': 'Failed to generate summary'}
    
    return app