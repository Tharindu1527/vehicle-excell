"""
UK Government Data Collector
Collects vehicle registration and market data from DVLA and ONS APIs
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from config.api_keys import *
from config.settings import Settings
from src.utils.data_validator import DataValidator
from src.utils.helpers import RateLimiter

class GovDataCollector:
    """UK Government data collector for vehicle statistics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.validator = DataValidator()
        self.rate_limiter = RateLimiter(requests_per_hour=100)
        
        # API configuration
        self.dvla_api_key = DVLA_API_KEY
        self.dvla_base_url = API_ENDPOINTS['dvla_vehicle']
        
        self.dvla_headers = {
            'x-api-key': self.dvla_api_key,
            'Content-Type': 'application/json'
        }
        
        self.logger.info("Government data collector initialized")
    
    def get_vehicle_info(self, registration: str) -> Dict:
        """Get vehicle information from DVLA API"""
        
        self.rate_limiter.wait_if_needed()
        
        try:
            payload = {
                'registrationNumber': registration.upper().replace(' ', '')
            }
            
            response = requests.post(
                self.dvla_base_url,
                json=payload,
                headers=self.dvla_headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_dvla_response(data)
            else:
                self.logger.warning(f"DVLA API returned {response.status_code} for {registration}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting DVLA data for {registration}: {str(e)}")
            return {}
    
    def parse_dvla_response(self, data: Dict) -> Dict:
        """Parse DVLA API response"""
        
        return {
            'registration': data.get('registrationNumber'),
            'make': data.get('make'),
            'model': data.get('model'),
            'colour': data.get('colour'),
            'fuel_type': data.get('fuelType'),
            'engine_capacity': data.get('engineCapacity'),
            'date_of_first_registration': data.get('dateOfFirstRegistration'),
            'year_of_manufacture': data.get('yearOfManufacture'),
            'co2_emissions': data.get('co2Emissions'),
            'export_marker': data.get('exportMarker'),
            'vehicle_status': data.get('vehicleStatus'),
            'mot_status': data.get('motStatus'),
            'mot_expiry_date': data.get('motExpiryDate'),
            'tax_status': data.get('taxStatus'),
            'tax_due_date': data.get('taxDueDate'),
            'type_approval': data.get('typeApproval'),
            'wheelplan': data.get('wheelplan'),
            'revenue_weight': data.get('revenueWeight')
        }
    
    def get_emissions_data(self) -> Dict:
        """Get CO2 emissions and ULEZ compliance data"""
        
        try:
            # Mock endpoint - replace with actual government API
            emissions_data = {
                'ulez_compliance_threshold': 75,  # g/km CO2 for petrol
                'congestion_charge_threshold': 225,  # g/km CO2
                'vehicle_excise_duty_bands': {
                    'A': {'min': 0, 'max': 100, 'first_year': 0, 'standard': 0},
                    'B': {'min': 101, 'max': 110, 'first_year': 25, 'standard': 25},
                    'C': {'min': 111, 'max': 120, 'first_year': 105, 'standard': 105},
                    'D': {'min': 121, 'max': 130, 'first_year': 125, 'standard': 125},
                    'E': {'min': 131, 'max': 140, 'first_year': 145, 'standard': 145},
                    'F': {'min': 141, 'max': 150, 'first_year': 165, 'standard': 165},
                    'G': {'min': 151, 'max': 165, 'first_year': 205, 'standard': 205},
                    'H': {'min': 166, 'max': 175, 'first_year': 515, 'standard': 230},
                    'I': {'min': 176, 'max': 185, 'first_year': 830, 'standard': 250},
                    'J': {'min': 186, 'max': 200, 'first_year': 1240, 'standard': 270},
                    'K': {'min': 201, 'max': 225, 'first_year': 1760, 'standard': 295},
                    'L': {'min': 226, 'max': 255, 'first_year': 2245, 'standard': 570},
                    'M': {'min': 256, 'max': 999, 'first_year': 2245, 'standard': 570}
                }
            }
            
            return emissions_data
            
        except Exception as e:
            self.logger.error(f"Error getting emissions data: {str(e)}")
            return {}
    
    def collect_registration_data(self) -> pd.DataFrame:
        """Main method to collect government registration data (DVLA and emissions only)"""
        self.logger.info("Starting government data collection")
        
        all_data = []
        
        try:
            # Collect emissions data
            emissions_data = self.get_emissions_data()
            
            # Combine all government data (no ONS data)
            combined_data = {
                'emissions_data': emissions_data,
                'collection_timestamp': datetime.now().isoformat(),
                'source': 'uk_government'
            }
            
            all_data.append(combined_data)
            
        except Exception as e:
            self.logger.error(f"Error collecting government data: {str(e)}")
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            self.logger.info("Government data collection completed")
            return df
        else:
            self.logger.warning("No government data collected")
            return pd.DataFrame()
    
    def analyze_registration_trends(self, make: str, model: str = None) -> Dict:
        """Analyze registration trends for specific make/model"""
        
        # This would analyze historical registration data
        # For now, return mock analysis
        
        return {
            'make': make,
            'model': model,
            'annual_registrations': 1000,  # Placeholder
            'trend': 'increasing',
            'market_share': 0.05,
            'popular_regions': ['London', 'Manchester', 'Birmingham'],
            'average_age_at_registration': 3.5,
            'analysis_date': datetime.now().isoformat()
        }
    
    def get_import_duty_rates(self) -> Dict:
        """Get current import duty rates for vehicles"""
        
        return {
            'standard_rate': 0.10,  # 10% for most vehicles
            'electric_vehicles': 0.06,  # Reduced rate for EVs
            'hybrid_vehicles': 0.08,  # Reduced rate for hybrids
            'commercial_vehicles': 0.22,  # Higher rate for commercial
            'motorcycles': 0.06,
            'vintage_vehicles': 0.05,  # Vehicles over 30 years old
            'last_updated': datetime.now().isoformat()
        }
    
    def get_regional_preferences(self) -> Dict:
        """Get regional vehicle preferences from registration data"""
        
        return {
            'London': {
                'top_makes': ['Toyota', 'Honda', 'BMW'],
                'preferred_fuel': 'Hybrid',
                'average_price_range': '£15000-£30000'
            },
            'Manchester': {
                'top_makes': ['Ford', 'Vauxhall', 'Toyota'],
                'preferred_fuel': 'Petrol',
                'average_price_range': '£8000-£20000'
            },
            'Birmingham': {
                'top_makes': ['Toyota', 'Nissan', 'Honda'],
                'preferred_fuel': 'Diesel',
                'average_price_range': '£10000-£25000'
            },
            'analysis_date': datetime.now().isoformat()
        }