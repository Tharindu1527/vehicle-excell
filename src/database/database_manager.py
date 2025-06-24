"""
Properly Fixed Database Manager with correct schema and error handling
"""

import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

from config.settings import Settings

class DatabaseManager:
    """Properly fixed database manager with correct schema"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.db_path = self.settings.DATABASE_PATH
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.logger.info(f"Database manager initialized with path: {self.db_path}")
    
    def initialize_database(self):
        """Initialize database with CORRECT schemas that match the data"""
        self.logger.info("Initializing database tables")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # UK market data table - MATCHES eBay collector output
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS uk_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT UNIQUE,
                    title TEXT,
                    make TEXT,
                    model TEXT,
                    year INTEGER,
                    price REAL,
                    currency TEXT,
                    mileage REAL,
                    fuel_type TEXT,
                    transmission TEXT,
                    body_type TEXT,
                    condition TEXT,
                    url TEXT,
                    image_url TEXT,
                    seller TEXT,
                    location TEXT,
                    shipping_cost REAL,
                    age INTEGER,
                    source TEXT,
                    collection_timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Japan auction data table - MATCHES Japan collector output
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS japan_auction_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    auction_id TEXT UNIQUE,
                    title TEXT,
                    make TEXT,
                    model TEXT,
                    year INTEGER,
                    mileage_km REAL,
                    mileage_miles REAL,
                    final_price_jpy REAL,
                    final_price_gbp REAL,
                    auction_house TEXT,
                    auction_date TEXT,
                    grade TEXT,
                    grade_score REAL,
                    fuel_type TEXT,
                    transmission TEXT,
                    body_type TEXT,
                    colour TEXT,
                    estimated_import_cost REAL,
                    shipping_cost REAL,
                    import_duty REAL,
                    vat REAL,
                    other_costs REAL,
                    total_landed_cost REAL,
                    source TEXT,
                    collection_timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Government data table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS government_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT,
                    data_content TEXT,
                    collection_timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Analysis results table - MATCHES profitability analyzer output
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    make TEXT,
                    model TEXT,
                    year INTEGER,
                    year_range TEXT,
                    uk_sample_size INTEGER,
                    japan_sample_size INTEGER,
                    match_type TEXT,
                    uk_avg_price REAL,
                    uk_median_price REAL,
                    uk_min_price REAL,
                    uk_max_price REAL,
                    uk_price_std REAL,
                    uk_avg_mileage REAL,
                    uk_median_mileage REAL,
                    uk_listings_count INTEGER,
                    japan_avg_auction_price REAL,
                    japan_median_auction_price REAL,
                    japan_min_auction_price REAL,
                    japan_max_auction_price REAL,
                    japan_avg_import_cost REAL,
                    japan_avg_total_cost REAL,
                    japan_median_total_cost REAL,
                    japan_avg_mileage REAL,
                    japan_avg_grade REAL,
                    japan_auctions_count INTEGER,
                    gross_profit REAL,
                    profit_margin REAL,
                    profit_margin_conservative REAL,
                    roi REAL,
                    price_volatility_uk REAL,
                    risk_score REAL,
                    market_share REAL,
                    listings_density INTEGER,
                    price_trend TEXT,
                    demand_score REAL,
                    avg_days_listed REAL,
                    profitability_score REAL,
                    market_demand_score REAL,
                    risk_assessment_score REAL,
                    liquidity_score REAL,
                    market_trends_score REAL,
                    final_score REAL,
                    final_score_percentile REAL,
                    score_grade TEXT,
                    investment_category TEXT,
                    recommendation TEXT,
                    overall_score REAL,
                    analysis_timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_uk_make_model_year ON uk_market_data(make, model, year)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_japan_make_model_year ON japan_auction_data(make, model, year)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_score ON analysis_results(final_score DESC)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection_timestamp ON uk_market_data(collection_timestamp)')
                
                conn.commit()
                self.logger.info("Database initialization completed")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def store_uk_data(self, uk_data: pd.DataFrame):
        """Store UK market data with proper error handling"""
        if uk_data.empty:
            self.logger.warning("No UK data to store")
            return
        
        try:
            # Get expected columns from database schema
            expected_columns = [
                'item_id', 'title', 'make', 'model', 'year', 'price', 'currency',
                'mileage', 'fuel_type', 'transmission', 'body_type', 'condition',
                'url', 'image_url', 'seller', 'location', 'shipping_cost', 'age',
                'source', 'collection_timestamp'
            ]
            
            # Ensure all expected columns exist
            uk_clean = uk_data.copy()
            for col in expected_columns:
                if col not in uk_clean.columns:
                    # Set appropriate defaults
                    if col in ['price', 'year', 'mileage', 'shipping_cost', 'age']:
                        uk_clean[col] = 0
                    elif col == 'currency':
                        uk_clean[col] = 'GBP'
                    elif col == 'collection_timestamp':
                        uk_clean[col] = datetime.now().isoformat()
                    else:
                        uk_clean[col] = None
            
            # Select only expected columns
            uk_clean = uk_clean[expected_columns]
            
            # Clean data types
            uk_clean['price'] = pd.to_numeric(uk_clean['price'], errors='coerce').fillna(0)
            uk_clean['year'] = pd.to_numeric(uk_clean['year'], errors='coerce').fillna(2018)
            uk_clean['mileage'] = pd.to_numeric(uk_clean['mileage'], errors='coerce')
            uk_clean['shipping_cost'] = pd.to_numeric(uk_clean['shipping_cost'], errors='coerce').fillna(0)
            uk_clean['age'] = pd.to_numeric(uk_clean['age'], errors='coerce').fillna(5)
            
            with sqlite3.connect(self.db_path) as conn:
                uk_clean.to_sql('uk_market_data', conn, if_exists='append', index=False)
                
            self.logger.info(f"Stored {len(uk_clean)} UK market records")
                
        except Exception as e:
            self.logger.error(f"Error storing UK data: {str(e)}")
            # Log detailed error info
            if not uk_data.empty:
                self.logger.error(f"UK data columns: {list(uk_data.columns)}")
                self.logger.error(f"UK data shape: {uk_data.shape}")
    
    def store_japan_data(self, japan_data: pd.DataFrame):
        """Store Japan auction data with proper error handling"""
        if japan_data.empty:
            self.logger.warning("No Japan data to store")
            return
        
        try:
            # Get expected columns from database schema
            expected_columns = [
                'auction_id', 'title', 'make', 'model', 'year', 'mileage_km', 'mileage_miles',
                'final_price_jpy', 'final_price_gbp', 'auction_house', 'auction_date',
                'grade', 'grade_score', 'fuel_type', 'transmission', 'body_type', 'colour',
                'estimated_import_cost', 'shipping_cost', 'import_duty', 'vat', 'other_costs',
                'total_landed_cost', 'source', 'collection_timestamp'
            ]
            
            # Ensure all expected columns exist
            japan_clean = japan_data.copy()
            for col in expected_columns:
                if col not in japan_clean.columns:
                    # Set appropriate defaults
                    if col in ['final_price_jpy', 'final_price_gbp', 'total_landed_cost', 'estimated_import_cost']:
                        japan_clean[col] = 0
                    elif col == 'grade_score':
                        japan_clean[col] = 5.0
                    elif col == 'collection_timestamp':
                        japan_clean[col] = datetime.now().isoformat()
                    else:
                        japan_clean[col] = None
            
            # Select only expected columns
            japan_clean = japan_clean[expected_columns]
            
            # Clean data types
            numeric_columns = ['year', 'mileage_km', 'mileage_miles', 'final_price_jpy', 'final_price_gbp',
                             'grade_score', 'estimated_import_cost', 'shipping_cost', 'import_duty',
                             'vat', 'other_costs', 'total_landed_cost']
            
            for col in numeric_columns:
                if col in japan_clean.columns:
                    japan_clean[col] = pd.to_numeric(japan_clean[col], errors='coerce')
            
            # Fill critical missing values
            japan_clean['final_price_gbp'] = japan_clean['final_price_gbp'].fillna(
                japan_clean['final_price_jpy'] * 0.0055 if 'final_price_jpy' in japan_clean.columns else 12000
            )
            japan_clean['total_landed_cost'] = japan_clean['total_landed_cost'].fillna(
                japan_clean['final_price_gbp'] * 1.35
            )
            
            with sqlite3.connect(self.db_path) as conn:
                japan_clean.to_sql('japan_auction_data', conn, if_exists='append', index=False)
                
            self.logger.info(f"Stored {len(japan_clean)} Japan auction records")
                
        except Exception as e:
            self.logger.error(f"Error storing Japan data: {str(e)}")
            if not japan_data.empty:
                self.logger.error(f"Japan data columns: {list(japan_data.columns)}")
                self.logger.error(f"Japan data shape: {japan_data.shape}")
    
    def store_analysis_results(self, profitability_results: pd.DataFrame, scores: pd.DataFrame):
        """Store analysis results with proper schema matching"""
        if scores.empty:
            self.logger.warning("No analysis results to store")
            return
        
        try:
            # Get expected columns from database schema
            expected_columns = [
                'make', 'model', 'year', 'year_range', 'uk_sample_size', 'japan_sample_size',
                'match_type', 'uk_avg_price', 'uk_median_price', 'uk_min_price', 'uk_max_price',
                'uk_price_std', 'uk_avg_mileage', 'uk_median_mileage', 'uk_listings_count',
                'japan_avg_auction_price', 'japan_median_auction_price', 'japan_min_auction_price',
                'japan_max_auction_price', 'japan_avg_import_cost', 'japan_avg_total_cost',
                'japan_median_total_cost', 'japan_avg_mileage', 'japan_avg_grade', 'japan_auctions_count',
                'gross_profit', 'profit_margin', 'profit_margin_conservative', 'roi',
                'price_volatility_uk', 'risk_score', 'market_share', 'listings_density',
                'price_trend', 'demand_score', 'avg_days_listed', 'profitability_score',
                'market_demand_score', 'risk_assessment_score', 'liquidity_score',
                'market_trends_score', 'final_score', 'final_score_percentile', 'score_grade',
                'investment_category', 'recommendation', 'overall_score', 'analysis_timestamp'
            ]
            
            # Prepare data for storage
            results_clean = scores.copy()
            
            # Ensure all expected columns exist
            for col in expected_columns:
                if col not in results_clean.columns:
                    # Set appropriate defaults
                    if col == 'analysis_timestamp':
                        results_clean[col] = datetime.now().isoformat()
                    elif col in ['final_score', 'overall_score', 'profitability_score', 'market_demand_score', 
                               'risk_assessment_score', 'liquidity_score', 'market_trends_score']:
                        results_clean[col] = 50.0  # Default score
                    elif col == 'score_grade':
                        results_clean[col] = 'C'
                    elif col == 'investment_category':
                        results_clean[col] = 'Moderate Opportunity'
                    elif col == 'recommendation':
                        results_clean[col] = 'Analyze Further'
                    elif col == 'match_type':
                        results_clean[col] = 'exact'
                    elif col == 'price_trend':
                        results_clean[col] = 'stable'
                    else:
                        results_clean[col] = None
            
            # Select only expected columns
            results_clean = results_clean[expected_columns]
            
            # Ensure numeric columns are properly typed
            numeric_columns = [col for col in expected_columns if col not in 
                             ['make', 'model', 'year_range', 'match_type', 'price_trend', 'score_grade', 
                              'investment_category', 'recommendation', 'analysis_timestamp']]
            
            for col in numeric_columns:
                if col in results_clean.columns:
                    results_clean[col] = pd.to_numeric(results_clean[col], errors='coerce')
            
            with sqlite3.connect(self.db_path) as conn:
                results_clean.to_sql('analysis_results', conn, if_exists='append', index=False)
                
            self.logger.info(f"Stored {len(results_clean)} analysis results")
                
        except Exception as e:
            self.logger.error(f"Error storing analysis results: {str(e)}")
            if not scores.empty:
                self.logger.error(f"Scores columns: {list(scores.columns)}")
                self.logger.error(f"Scores shape: {scores.shape}")
    
    def store_gov_data(self, gov_data: pd.DataFrame):
        """Store government data"""
        if gov_data.empty:
            self.logger.warning("No government data to store")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for _, row in gov_data.iterrows():
                    data_content = json.dumps(row.to_dict())
                    
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO government_data (data_type, data_content, collection_timestamp)
                        VALUES (?, ?, ?)
                    ''', ('registration_stats', data_content, datetime.now().isoformat()))
                
                conn.commit()
                self.logger.info(f"Stored {len(gov_data)} government data records")
                
        except Exception as e:
            self.logger.error(f"Error storing government data: {str(e)}")
    
    def get_uk_data(self, days_back: int = 7) -> pd.DataFrame:
        """Get UK market data from last N days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM uk_market_data 
                    WHERE collection_timestamp >= ?
                    ORDER BY collection_timestamp DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=(cutoff_date,))
                
                self.logger.info(f"Retrieved {len(df)} UK market records")
                return df
                
        except Exception as e:
            self.logger.error(f"Error retrieving UK data: {str(e)}")
            return pd.DataFrame()
    
    def get_japan_data(self, days_back: int = 7) -> pd.DataFrame:
        """Get Japan auction data from last N days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM japan_auction_data 
                    WHERE collection_timestamp >= ?
                    ORDER BY collection_timestamp DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=(cutoff_date,))
                
                self.logger.info(f"Retrieved {len(df)} Japan auction records")
                return df
                
        except Exception as e:
            self.logger.error(f"Error retrieving Japan data: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_analysis_results(self, limit: int = 100) -> pd.DataFrame:
        """Get latest analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM analysis_results 
                    ORDER BY final_score DESC, created_at DESC
                    LIMIT ?
                '''
                
                df = pd.read_sql_query(query, conn, params=(limit,))
                
                self.logger.info(f"Retrieved {len(df)} analysis results")
                return df
                
        except Exception as e:
            self.logger.error(f"Error retrieving analysis results: {str(e)}")
            return pd.DataFrame()
    
    def get_market_summary(self) -> Dict:
        """Get market summary statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # UK market summary
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_listings,
                        AVG(price) as avg_price,
                        MIN(price) as min_price,
                        MAX(price) as max_price,
                        COUNT(DISTINCT make) as unique_makes
                    FROM uk_market_data 
                    WHERE collection_timestamp >= ?
                ''', ((datetime.now() - timedelta(days=7)).isoformat(),))
                
                uk_summary = cursor.fetchone()
                
                # Japan auction summary
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_auctions,
                        AVG(final_price_gbp) as avg_price,
                        MIN(final_price_gbp) as min_price,
                        MAX(final_price_gbp) as max_price,
                        AVG(total_landed_cost) as avg_total_cost
                    FROM japan_auction_data 
                    WHERE collection_timestamp >= ?
                ''', ((datetime.now() - timedelta(days=7)).isoformat(),))
                
                japan_summary = cursor.fetchone()
                
                # Analysis summary
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_analyzed,
                        AVG(profit_margin) as avg_profit_margin,
                        MAX(profit_margin) as max_profit_margin,
                        AVG(final_score) as avg_score,
                        MAX(final_score) as max_score,
                        SUM(gross_profit) as total_profit_potential
                    FROM analysis_results 
                    WHERE analysis_timestamp >= ?
                ''', ((datetime.now() - timedelta(days=7)).isoformat(),))
                
                analysis_summary = cursor.fetchone()
                
                return {
                    'uk_market': {
                        'total_listings': uk_summary[0] if uk_summary else 0,
                        'avg_price': uk_summary[1] if uk_summary else 0,
                        'price_range': f"£{uk_summary[2]:,.0f} - £{uk_summary[3]:,.0f}" if uk_summary and uk_summary[2] else "N/A",
                        'unique_makes': uk_summary[4] if uk_summary else 0
                    },
                    'japan_market': {
                        'total_auctions': japan_summary[0] if japan_summary else 0,
                        'avg_auction_price': japan_summary[1] if japan_summary else 0,
                        'avg_total_cost': japan_summary[4] if japan_summary else 0,
                        'price_range': f"£{japan_summary[2]:,.0f} - £{japan_summary[3]:,.0f}" if japan_summary and japan_summary[2] else "N/A"
                    },
                    'analysis': {
                        'vehicles_analyzed': analysis_summary[0] if analysis_summary else 0,
                        'avg_profit_margin': analysis_summary[1] if analysis_summary and analysis_summary[1] else 0,
                        'max_profit_margin': analysis_summary[2] if analysis_summary and analysis_summary[2] else 0,
                        'avg_score': round(analysis_summary[3], 1) if analysis_summary and analysis_summary[3] else 0,
                        'max_score': round(analysis_summary[4], 1) if analysis_summary and analysis_summary[4] else 0,
                        'total_profit_potential': analysis_summary[5] if analysis_summary and analysis_summary[5] else 0
                    },
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return {}
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                tables = ['uk_market_data', 'japan_auction_data', 'government_data', 'analysis_results']
                stats = {}
                
                for table in tables:
                    try:
                        cursor.execute(f'SELECT COUNT(*) FROM {table}')
                        count = cursor.fetchone()[0]
                        stats[table] = count
                    except:
                        stats[table] = 0
                
                # Database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                stats['database_size_mb'] = round(db_size / (1024 * 1024), 2)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond retention period"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean old UK data
                cursor.execute('DELETE FROM uk_market_data WHERE collection_timestamp < ?', (cutoff_date,))
                uk_deleted = cursor.rowcount
                
                # Clean old Japan data
                cursor.execute('DELETE FROM japan_auction_data WHERE collection_timestamp < ?', (cutoff_date,))
                japan_deleted = cursor.rowcount
                
                # Clean old analysis results
                cursor.execute('DELETE FROM analysis_results WHERE analysis_timestamp < ?', (cutoff_date,))
                analysis_deleted = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Cleanup completed: UK:{uk_deleted}, Japan:{japan_deleted}, Analysis:{analysis_deleted}")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")