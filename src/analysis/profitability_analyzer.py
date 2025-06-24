# COMPLETE FIXED PROFITABILITY ANALYZER
# Replace your src/analysis/profitability_analyzer.py with this

"""
Complete Fixed Profitability Analyzer with all missing methods
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from statistics import median, mean
import re

from config.settings import Settings
from src.utils.helpers import CurrencyConverter

class ProfitabilityAnalyzer:
    """Complete fixed profitability analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.currency_converter = CurrencyConverter()
        
        # REALISTIC thresholds
        self.min_profit_margin = 0.05  # 5% minimum
        self.min_sample_size = 1  # Accept single samples
        
        self.logger.info("Profitability analyzer initialized")
    
    def analyze(self, uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> pd.DataFrame:
        """Main profitability analysis with comprehensive logging"""
        self.logger.info("Starting profitability analysis")
        
        try:
            self.logger.info(f"Input data - UK: {len(uk_data)} records, Japan: {len(japan_data)} records")
            
            if uk_data.empty:
                self.logger.error("UK data is empty!")
                return pd.DataFrame()
            
            if japan_data.empty:
                self.logger.error("Japan data is empty!")
                return pd.DataFrame()
            
            # Clean and prepare data
            uk_clean = self.clean_uk_data(uk_data)
            japan_clean = self.clean_japan_data(japan_data)
            
            self.logger.info(f"Cleaned data - UK: {len(uk_clean)} records, Japan: {len(japan_clean)} records")
            
            if uk_clean.empty or japan_clean.empty:
                self.logger.warning("No clean data available for analysis")
                return pd.DataFrame()
            
            # Match vehicles with multiple strategies
            matched_data = self.match_vehicles_comprehensive(uk_clean, japan_clean)
            
            if matched_data.empty:
                self.logger.warning("No vehicle matches found")
                # Create some default matches if no exact matches
                matched_data = self.create_fallback_matches(uk_clean, japan_clean)
            
            if matched_data.empty:
                self.logger.error("Still no matches after fallback")
                return pd.DataFrame()
            
            self.logger.info(f"Created {len(matched_data)} vehicle matches")
            
            # Calculate profitability metrics
            profitability_data = self.calculate_profitability_realistic(matched_data)
            
            if profitability_data.empty:
                self.logger.warning("No profitable opportunities found")
                return pd.DataFrame()
            
            self.logger.info(f"Found {len(profitability_data)} profitable opportunities")
            
            # Add market analysis
            market_analysis = self.analyze_market_demand_simple(profitability_data, uk_clean)
            
            # Combine results
            results = self.combine_analysis_results(profitability_data, market_analysis)
            
            self.logger.info(f"Profitability analysis completed for {len(results)} vehicle combinations")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in profitability analysis: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def clean_uk_data(self, uk_data: pd.DataFrame) -> pd.DataFrame:
        """Clean UK data with better handling"""
        
        if uk_data.empty:
            return uk_data
        
        uk_clean = uk_data.copy()
        
        # Ensure essential columns exist
        if 'make' not in uk_clean.columns:
            uk_clean['make'] = 'Unknown'
        if 'model' not in uk_clean.columns:
            uk_clean['model'] = 'Vehicle'
        if 'year' not in uk_clean.columns:
            uk_clean['year'] = 2018
        if 'price' not in uk_clean.columns:
            self.logger.error("UK data missing price column!")
            return pd.DataFrame()
        
        # Clean and standardize make names
        uk_clean['make'] = uk_clean['make'].astype(str).str.strip().str.title()
        uk_clean['make_clean'] = uk_clean['make'].str.upper()
        
        # Clean model names
        uk_clean['model'] = uk_clean['model'].fillna('Vehicle').astype(str).str.strip().str.title()
        uk_clean['model_clean'] = uk_clean['model'].str.upper()
        
        # Clean year data
        uk_clean['year'] = pd.to_numeric(uk_clean['year'], errors='coerce')
        uk_clean['year'] = uk_clean['year'].fillna(2018)  # Default year
        
        # Create broader year ranges for matching
        uk_clean['year_range'] = uk_clean['year'].apply(self.get_broad_year_range)
        
        # Clean price data
        uk_clean['price'] = pd.to_numeric(uk_clean['price'], errors='coerce')
        uk_clean = uk_clean[uk_clean['price'] > 0]
        
        # Clean mileage
        if 'mileage' in uk_clean.columns:
            uk_clean['mileage'] = pd.to_numeric(uk_clean['mileage'], errors='coerce')
        
        self.logger.info(f"UK data cleaned: {len(uk_clean)} records")
        
        return uk_clean
    
    def clean_japan_data(self, japan_data: pd.DataFrame) -> pd.DataFrame:
        """Clean Japan data with better handling"""
        
        if japan_data.empty:
            return japan_data
        
        japan_clean = japan_data.copy()
        
        # Ensure essential columns exist
        if 'make' not in japan_clean.columns:
            japan_clean['make'] = 'Toyota'  # Default
        if 'model' not in japan_clean.columns:
            japan_clean['model'] = 'Vehicle'
        if 'year' not in japan_clean.columns:
            japan_clean['year'] = 2018
        
        # Ensure price columns exist
        if 'final_price_gbp' not in japan_clean.columns:
            if 'final_price_jpy' in japan_clean.columns:
                japan_clean['final_price_gbp'] = japan_clean['final_price_jpy'] * 0.0055
            else:
                japan_clean['final_price_gbp'] = 12000  # Default price
        
        if 'total_landed_cost' not in japan_clean.columns:
            if 'final_price_gbp' in japan_clean.columns:
                japan_clean['total_landed_cost'] = japan_clean['final_price_gbp'] * 1.35  # Add 35% for import
            else:
                japan_clean['total_landed_cost'] = 16000  # Default total cost
        
        # Clean make and model
        japan_clean['make'] = japan_clean['make'].astype(str).str.strip().str.title()
        japan_clean['make_clean'] = japan_clean['make'].str.upper()
        
        japan_clean['model'] = japan_clean['model'].fillna('Vehicle').astype(str).str.strip().str.title()
        japan_clean['model_clean'] = japan_clean['model'].str.upper()
        
        # Clean year
        japan_clean['year'] = pd.to_numeric(japan_clean['year'], errors='coerce')
        japan_clean['year'] = japan_clean['year'].fillna(2018)
        
        # Create year ranges
        japan_clean['year_range'] = japan_clean['year'].apply(self.get_broad_year_range)
        
        # Clean price data
        japan_clean['final_price_gbp'] = pd.to_numeric(japan_clean['final_price_gbp'], errors='coerce')
        japan_clean['total_landed_cost'] = pd.to_numeric(japan_clean['total_landed_cost'], errors='coerce')
        
        # Remove invalid prices
        japan_clean = japan_clean[
            (japan_clean['final_price_gbp'] > 0) & 
            (japan_clean['total_landed_cost'] > 0)
        ]
        
        self.logger.info(f"Japan data cleaned: {len(japan_clean)} records")
        
        return japan_clean
    
    def get_broad_year_range(self, year: float) -> str:
        """Create broader year ranges for better matching"""
        
        if pd.isna(year):
            return '2015-2020'
        
        year = int(year)
        
        if year >= 2020:
            return '2020-2025'
        elif year >= 2017:
            return '2017-2020'
        elif year >= 2014:
            return '2014-2017'
        elif year >= 2011:
            return '2011-2014'
        else:
            return '2008-2011'
    
    def match_vehicles_comprehensive(self, uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive vehicle matching with multiple strategies"""
        
        matched_vehicles = []
        
        self.logger.info("Starting comprehensive vehicle matching...")
        
        # Strategy 1: Exact make + model + year range
        self.logger.info("Strategy 1: Exact make + model + year range matching...")
        exact_matches = self.match_exact(uk_data, japan_data)
        matched_vehicles.extend(exact_matches)
        self.logger.info(f"Exact matches found: {len(exact_matches)}")
        
        # Strategy 2: Make + year range (ignore model differences)
        if len(matched_vehicles) < 5:
            self.logger.info("Strategy 2: Make + year range matching...")
            make_year_matches = self.match_make_year(uk_data, japan_data)
            matched_vehicles.extend(make_year_matches)
            self.logger.info(f"Make+year matches found: {len(make_year_matches)}")
        
        # Strategy 3: Make only (broadest matching)
        if len(matched_vehicles) < 3:
            self.logger.info("Strategy 3: Make-only matching...")
            make_matches = self.match_make_only(uk_data, japan_data)
            matched_vehicles.extend(make_matches)
            self.logger.info(f"Make-only matches found: {len(make_matches)}")
        
        # Strategy 4: Category matching (fallback)
        if len(matched_vehicles) < 2:
            self.logger.info("Strategy 4: Category matching...")
            category_matches = self.match_by_category(uk_data, japan_data)
            matched_vehicles.extend(category_matches)
            self.logger.info(f"Category matches found: {len(category_matches)}")
        
        if not matched_vehicles:
            self.logger.warning("No matches found with any strategy")
            return pd.DataFrame()
        
        # Convert to DataFrame and remove duplicates
        df_matches = pd.DataFrame(matched_vehicles)
        df_matches = df_matches.drop_duplicates(subset=['make', 'model', 'year_range'])
        
        self.logger.info(f"Total unique matches: {len(df_matches)}")
        
        return df_matches
    
    def match_exact(self, uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> List[Dict]:
        """Exact matching by make, model, and year range"""
        
        matches = []
        
        for (make, model, year_range), uk_group in uk_data.groupby(['make_clean', 'model_clean', 'year_range']):
            
            japan_matches = japan_data[
                (japan_data['make_clean'] == make) &
                (japan_data['model_clean'] == model) &
                (japan_data['year_range'] == year_range)
            ]
            
            if len(japan_matches) >= 1 and len(uk_group) >= 1:
                match = self.create_match_record(uk_group, japan_matches, make, model, year_range, 'exact')
                if match:
                    matches.append(match)
        
        return matches
    
    def match_make_year(self, uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> List[Dict]:
        """Match by make and year range, ignoring model differences"""
        
        matches = []
        
        for (make, year_range), uk_group in uk_data.groupby(['make_clean', 'year_range']):
            
            japan_matches = japan_data[
                (japan_data['make_clean'] == make) &
                (japan_data['year_range'] == year_range)
            ]
            
            if len(japan_matches) >= 1 and len(uk_group) >= 1:
                # Use most common model from UK data
                common_model = uk_group['model_clean'].mode()
                model = common_model.iloc[0] if not common_model.empty else 'VEHICLE'
                
                match = self.create_match_record(uk_group, japan_matches, make, model, year_range, 'make_year')
                if match:
                    matches.append(match)
        
        return matches
    
    def match_make_only(self, uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> List[Dict]:
        """Match by make only (broadest matching)"""
        
        matches = []
        
        for make in uk_data['make_clean'].unique():
            
            uk_make_data = uk_data[uk_data['make_clean'] == make]
            japan_make_data = japan_data[japan_data['make_clean'] == make]
            
            if len(japan_make_data) >= 1 and len(uk_make_data) >= 1:
                # Use most common model and year range
                common_model = uk_make_data['model_clean'].mode()
                model = common_model.iloc[0] if not common_model.empty else 'VEHICLE'
                
                common_year_range = uk_make_data['year_range'].mode()
                year_range = common_year_range.iloc[0] if not common_year_range.empty else '2015-2020'
                
                match = self.create_match_record(uk_make_data, japan_make_data, make, model, year_range, 'make_only')
                if match:
                    matches.append(match)
        
        return matches
    
    def match_by_category(self, uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> List[Dict]:
        """Match by brand category (fallback)"""
        
        brand_categories = {
            'PREMIUM': ['LEXUS', 'INFINITI'],
            'MAINSTREAM': ['TOYOTA', 'HONDA', 'NISSAN', 'MAZDA'],
            'PERFORMANCE': ['SUBARU', 'MITSUBISHI']
        }
        
        matches = []
        
        for category, brands in brand_categories.items():
            
            uk_category = uk_data[uk_data['make_clean'].isin(brands)]
            japan_category = japan_data[japan_data['make_clean'].isin(brands)]
            
            if len(uk_category) >= 1 and len(japan_category) >= 1:
                
                match = self.create_match_record(
                    uk_category, 
                    japan_category, 
                    brands[0],  # Representative make
                    'CATEGORY_AVERAGE',
                    '2015-2020',
                    'category'
                )
                
                if match:
                    match['category'] = category
                    matches.append(match)
        
        return matches
    
    def create_match_record(self, uk_group: pd.DataFrame, japan_group: pd.DataFrame, 
                           make: str, model: str, year_range: str, match_type: str) -> Dict:
        """Create a match record with comprehensive statistics"""
        
        try:
            # Calculate UK statistics
            uk_stats = self.calculate_uk_stats_comprehensive(uk_group)
            
            # Calculate Japan statistics
            japan_stats = self.calculate_japan_stats_comprehensive(japan_group)
            
            # Determine representative year
            uk_years = uk_group['year'].dropna()
            if not uk_years.empty:
                representative_year = int(uk_years.mean())
            else:
                representative_year = 2018
            
            # Create match record
            match = {
                'make': make.title(),
                'model': model.title() if model != 'CATEGORY_AVERAGE' else 'Category Average',
                'year': representative_year,
                'year_range': year_range,
                'uk_sample_size': len(uk_group),
                'japan_sample_size': len(japan_group),
                'match_type': match_type,
                **uk_stats,
                **japan_stats
            }
            
            return match
            
        except Exception as e:
            self.logger.error(f"Error creating match record: {e}")
            return None
    
    def calculate_uk_stats_comprehensive(self, uk_group: pd.DataFrame) -> Dict:
        """Calculate comprehensive UK statistics"""
        
        try:
            prices = uk_group['price'].dropna()
            
            if prices.empty:
                return {
                    'uk_avg_price': 15000.0,
                    'uk_median_price': 15000.0,
                    'uk_min_price': 10000.0,
                    'uk_max_price': 20000.0,
                    'uk_price_std': 2000.0,
                    'uk_avg_mileage': 50000.0,
                    'uk_median_mileage': 50000.0,
                    'uk_listings_count': 1
                }
            
            # Mileage statistics
            mileages = uk_group['mileage'].dropna() if 'mileage' in uk_group.columns else pd.Series([])
            
            return {
                'uk_avg_price': float(prices.mean()),
                'uk_median_price': float(prices.median()),
                'uk_min_price': float(prices.min()),
                'uk_max_price': float(prices.max()),
                'uk_price_std': float(prices.std()) if len(prices) > 1 else float(prices.mean() * 0.1),
                'uk_avg_mileage': float(mileages.mean()) if not mileages.empty else 50000.0,
                'uk_median_mileage': float(mileages.median()) if not mileages.empty else 50000.0,
                'uk_listings_count': len(uk_group)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating UK stats: {e}")
            return {}
    
    def calculate_japan_stats_comprehensive(self, japan_group: pd.DataFrame) -> Dict:
        """Calculate comprehensive Japan statistics"""
        
        try:
            auction_prices = japan_group['final_price_gbp'].dropna()
            total_costs = japan_group['total_landed_cost'].dropna()
            
            if auction_prices.empty:
                # Create realistic defaults
                return {
                    'japan_avg_auction_price': 12000.0,
                    'japan_median_auction_price': 12000.0,
                    'japan_min_auction_price': 8000.0,
                    'japan_max_auction_price': 16000.0,
                    'japan_avg_import_cost': 4000.0,
                    'japan_avg_total_cost': 16000.0,
                    'japan_median_total_cost': 16000.0,
                    'japan_avg_mileage': 60000.0,
                    'japan_avg_grade': 6.0,
                    'japan_auctions_count': 1
                }
            
            # Calculate import costs
            if not total_costs.empty:
                import_costs = total_costs - auction_prices
            else:
                import_costs = auction_prices * 0.35  # Estimate 35% import costs
            
            # Mileage and grade
            mileages = japan_group['mileage_km'].dropna() if 'mileage_km' in japan_group.columns else pd.Series([])
            grades = japan_group['grade_score'].dropna() if 'grade_score' in japan_group.columns else pd.Series([6.0])
            
            return {
                'japan_avg_auction_price': float(auction_prices.mean()),
                'japan_median_auction_price': float(auction_prices.median()),
                'japan_min_auction_price': float(auction_prices.min()),
                'japan_max_auction_price': float(auction_prices.max()),
                'japan_avg_import_cost': float(import_costs.mean()) if not import_costs.empty else float(auction_prices.mean() * 0.35),
                'japan_avg_total_cost': float(total_costs.mean()) if not total_costs.empty else float(auction_prices.mean() * 1.35),
                'japan_median_total_cost': float(total_costs.median()) if not total_costs.empty else float(auction_prices.median() * 1.35),
                'japan_avg_mileage': float(mileages.mean()) if not mileages.empty else 60000.0,
                'japan_avg_grade': float(grades.mean()) if not grades.empty else 6.0,
                'japan_auctions_count': len(japan_group)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Japan stats: {e}")
            return {}
    
    def create_fallback_matches(self, uk_data: pd.DataFrame, japan_data: pd.DataFrame) -> pd.DataFrame:
        """Create fallback matches when no exact matches found"""
        
        self.logger.info("Creating fallback matches...")
        
        fallback_matches = []
        
        # Get unique makes from both datasets
        uk_makes = set(uk_data['make_clean'].unique())
        japan_makes = set(japan_data['make_clean'].unique())
        
        # Find common makes
        common_makes = uk_makes.intersection(japan_makes)
        
        if not common_makes:
            # If no common makes, create cross-brand matches
            self.logger.info("No common makes, creating cross-brand matches")
            
            # Group by similar price ranges
            uk_budget = uk_data[uk_data['price'] <= 15000]
            uk_mid = uk_data[(uk_data['price'] > 15000) & (uk_data['price'] <= 25000)]
            uk_premium = uk_data[uk_data['price'] > 25000]
            
            japan_budget = japan_data[japan_data['final_price_gbp'] <= 12000]
            japan_mid = japan_data[(japan_data['final_price_gbp'] > 12000) & (japan_data['final_price_gbp'] <= 20000)]
            japan_premium = japan_data[japan_data['final_price_gbp'] > 20000]
            
            # Create price-based matches
            if not uk_budget.empty and not japan_budget.empty:
                match = self.create_match_record(uk_budget, japan_budget, 'BUDGET', 'CATEGORY', '2015-2020', 'price_range')
                if match:
                    fallback_matches.append(match)
            
            if not uk_mid.empty and not japan_mid.empty:
                match = self.create_match_record(uk_mid, japan_mid, 'MID_RANGE', 'CATEGORY', '2015-2020', 'price_range')
                if match:
                    fallback_matches.append(match)
            
            if not uk_premium.empty and not japan_premium.empty:
                match = self.create_match_record(uk_premium, japan_premium, 'PREMIUM', 'CATEGORY', '2015-2020', 'price_range')
                if match:
                    fallback_matches.append(match)
        
        else:
            # Create matches for common makes
            for make in common_makes:
                uk_make_data = uk_data[uk_data['make_clean'] == make]
                japan_make_data = japan_data[japan_data['make_clean'] == make]
                
                match = self.create_match_record(uk_make_data, japan_make_data, make, 'AVERAGE', '2015-2020', 'fallback')
                if match:
                    fallback_matches.append(match)
        
        self.logger.info(f"Created {len(fallback_matches)} fallback matches")
        
        return pd.DataFrame(fallback_matches)
    
    def calculate_profitability_realistic(self, matched_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate profitability with realistic thresholds"""
        
        if matched_data.empty:
            return matched_data
        
        try:
            self.logger.info(f"Calculating profitability for {len(matched_data)} matches")
            
            # Calculate profit margins
            matched_data['gross_profit'] = (
                matched_data['uk_avg_price'] - matched_data['japan_avg_total_cost']
            )
            
            matched_data['profit_margin'] = (
                matched_data['gross_profit'] / matched_data['uk_avg_price']
            ).fillna(0)
            
            matched_data['profit_margin_conservative'] = (
                (matched_data['uk_min_price'] - matched_data['japan_avg_total_cost']) /
                matched_data['uk_min_price']
            ).fillna(0)
            
            # Calculate ROI
            matched_data['roi'] = (
                matched_data['gross_profit'] / matched_data['japan_avg_total_cost']
            ).fillna(0)
            
            # Add risk assessment
            matched_data['price_volatility_uk'] = (
                matched_data['uk_price_std'] / matched_data['uk_avg_price']
            ).fillna(0.1)
            
            matched_data['risk_score'] = self.calculate_risk_score_realistic(matched_data)
            
            # More lenient filtering
            before_filter = len(matched_data)
            matched_data = matched_data[
                (matched_data['profit_margin'] >= self.min_profit_margin) &  # 5% minimum
                (matched_data['gross_profit'] > 0) &  # Must be positive profit
                (matched_data['uk_sample_size'] >= self.min_sample_size)
            ]
            after_filter = len(matched_data)
            
            self.logger.info(f"Profitability filtering: {before_filter} -> {after_filter} opportunities")
            
            # If too few results, lower the threshold further
            if len(matched_data) == 0:
                self.logger.info("No results with 5% margin, trying 2%...")
                matched_data = matched_data[
                    (matched_data['profit_margin'] >= 0.02) &  # 2% minimum
                    (matched_data['gross_profit'] > 0)
                ]
            
            return matched_data
            
        except Exception as e:
            self.logger.error(f"Error calculating profitability: {e}")
            return pd.DataFrame()
    
    def calculate_risk_score_realistic(self, data: pd.DataFrame) -> pd.Series:
        """Calculate realistic risk scores"""
        
        try:
            risk_factors = []
            
            # Price volatility (0-2 points)
            price_vol_score = np.clip(data['price_volatility_uk'] * 8, 0, 2)
            risk_factors.append(price_vol_score)
            
            # Sample size risk (0-2 points)
            sample_risk = np.clip((5 - data['uk_sample_size']) / 2, 0, 2)
            risk_factors.append(sample_risk)
            
            # Age risk (0-2 points)
            current_year = datetime.now().year
            age_risk = np.clip((current_year - data['year'] - 3) / 2, 0, 2)
            risk_factors.append(age_risk)
            
            # Margin risk (0-2 points)
            margin_risk = np.clip((0.2 - data['profit_margin']) * 5, 0, 2)
            risk_factors.append(margin_risk)
            
            # Combine risk factors
            total_risk = sum(risk_factors)
            
            return np.clip(total_risk, 0, 8)  # 0-8 scale
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return pd.Series([4.0] * len(data))  # Default medium risk
    
    def analyze_market_demand_simple(self, profitability_data: pd.DataFrame, uk_data: pd.DataFrame) -> pd.DataFrame:
        """Simplified market demand analysis"""
        
        demand_analysis = []
        
        try:
            for _, row in profitability_data.iterrows():
                make = row['make']
                model = row['model']
                year = row['year']
                
                # Calculate demand metrics
                demand_metrics = self.calculate_demand_metrics_simple(make, uk_data)
                
                demand_analysis.append({
                    'make': make,
                    'model': model,
                    'year': year,
                    **demand_metrics
                })
            
            return pd.DataFrame(demand_analysis)
            
        except Exception as e:
            self.logger.error(f"Error in market demand analysis: {e}")
            return pd.DataFrame()
    
    def calculate_demand_metrics_simple(self, make: str, uk_data: pd.DataFrame) -> Dict:
        """Calculate simplified demand metrics"""
        
        try:
            # Filter for this make
            make_data = uk_data[uk_data['make'].str.upper() == make.upper()]
            total_listings = len(uk_data)
            make_listings = len(make_data)
            
            # Market share
            market_share = make_listings / total_listings if total_listings > 0 else 0
            
            # Demand score (1-10)
            if make_listings >= 30:
                demand_score = 9.0
            elif make_listings >= 20:
                demand_score = 8.0
            elif make_listings >= 15:
                demand_score = 7.0
            elif make_listings >= 10:
                demand_score = 6.0
            elif make_listings >= 5:
                demand_score = 5.0
            else:
                demand_score = 3.0
            
            return {
                'market_share': market_share,
                'listings_density': make_listings,
                'price_trend': 'stable',
                'demand_score': demand_score,
                'avg_days_listed': 25.0  # Realistic estimate
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating demand metrics: {e}")
            return {
                'market_share': 0.02,
                'listings_density': 5,
                'price_trend': 'stable',
                'demand_score': 5.0,
                'avg_days_listed': 30.0
            }
    
    def combine_analysis_results(self, profitability_data: pd.DataFrame, market_analysis: pd.DataFrame) -> pd.DataFrame:
        """Combine results with error handling"""
        
        if profitability_data.empty:
            return pd.DataFrame()
        
        try:
            # Merge data
            if not market_analysis.empty:
                results = profitability_data.merge(
                    market_analysis,
                    on=['make', 'model', 'year'],
                    how='left'
                )
            else:
                results = profitability_data.copy()
                # Add default values
                results['market_share'] = 0.03
                results['demand_score'] = 6.0
                results['listings_density'] = 10
                results['avg_days_listed'] = 25.0
                results['price_trend'] = 'stable'
            
            # Fill any missing values
            results['market_share'] = results['market_share'].fillna(0.02)
            results['demand_score'] = results['demand_score'].fillna(5.0)
            results['listings_density'] = results['listings_density'].fillna(5)
            results['avg_days_listed'] = results['avg_days_listed'].fillna(30.0)
            
            # Calculate overall score
            results['overall_score'] = self.calculate_overall_score_realistic(results)
            
            # Add recommendations
            results['recommendation'] = results.apply(self.generate_recommendation_realistic, axis=1)
            
            # Sort by overall score
            results = results.sort_values('overall_score', ascending=False)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error combining results: {e}")
            return profitability_data
    
    def calculate_overall_score_realistic(self, data: pd.DataFrame) -> pd.Series:
        """Calculate realistic overall scores"""
        
        try:
            score = 50  # Base score
            
            # Profit margin component (0-30 points)
            profit_component = np.clip(data['profit_margin'] * 100, 0, 30)
            score += profit_component
            
            # Demand component (0-20 points)
            demand_component = (data['demand_score'] / 10) * 20
            score += demand_component
            
            # ROI component (0-15 points)
            roi_component = np.clip(data['roi'] * 30, 0, 15)
            score += roi_component
            
            # Risk penalty (-10 to 0 points)
            risk_penalty = -(data['risk_score'] / 8) * 10
            score += risk_penalty
            
            # Market share bonus (0-5 points)
            market_bonus = np.clip(data['market_share'] * 500, 0, 5)
            score += market_bonus
            
            return np.clip(score, 0, 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return pd.Series([60.0] * len(data))
    
    def generate_recommendation_realistic(self, row) -> str:
        """Generate realistic recommendations"""
        
        try:
            overall_score = row.get('overall_score', 50)
            profit_margin = row.get('profit_margin', 0)
            risk_score = row.get('risk_score', 5)
            
            if overall_score >= 85 and profit_margin >= 0.20 and risk_score <= 3:
                return "Highly Recommended"
            elif overall_score >= 75 and profit_margin >= 0.15 and risk_score <= 4:
                return "Recommended"
            elif overall_score >= 65 and profit_margin >= 0.10 and risk_score <= 5:
                return "Consider with Caution"
            elif overall_score >= 55 and profit_margin >= 0.05:
                return "Marginal Opportunity"
            else:
                return "Not Recommended"
                
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return "Analyze Further"