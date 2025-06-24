"""
Data Validation Utilities
Validates vehicle data for accuracy and completeness
"""

import logging
from typing import Dict, Any
from datetime import datetime

class DataValidator:
    """Data validation utilities for vehicle data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_vehicle_data(self, vehicle_data: Dict[str, Any]) -> bool:
        """Validate vehicle data structure and content"""
        
        if not isinstance(vehicle_data, dict):
            return False
        
        # Required fields
        required_fields = ['make', 'price']
        
        # Check required fields exist and are not None
        for field in required_fields:
            if field not in vehicle_data or vehicle_data[field] is None:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate price
        try:
            price = float(vehicle_data['price'])
            if price <= 0 or price > 1000000:  # Reasonable price range
                self.logger.warning(f"Invalid price: {price}")
                return False
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid price format: {vehicle_data.get('price')}")
            return False
        
        # Validate year if present
        if 'year' in vehicle_data and vehicle_data['year'] is not None:
            try:
                year = int(vehicle_data['year'])
                current_year = datetime.now().year
                if year < 1990 or year > current_year + 1:
                    self.logger.warning(f"Invalid year: {year}")
                    return False
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid year format: {vehicle_data.get('year')}")
                return False
        
        # Validate mileage if present
        if 'mileage' in vehicle_data and vehicle_data['mileage'] is not None:
            try:
                mileage = float(vehicle_data['mileage'])
                if mileage < 0 or mileage > 500000:  # Reasonable mileage range
                    self.logger.warning(f"Invalid mileage: {mileage}")
                    return False
            except (ValueError, TypeError):
                # Mileage validation is not strict, just log warning
                self.logger.debug(f"Invalid mileage format: {vehicle_data.get('mileage')}")
        
        return True
    
    def validate_price_range(self, price: float, min_price: float = 500, max_price: float = 150000) -> bool:
        """Validate if price is within reasonable range"""
        return min_price <= price <= max_price
    
    def validate_year_range(self, year: int, max_age: int = 20) -> bool:
        """Validate if vehicle year is within acceptable range"""
        current_year = datetime.now().year
        return (current_year - max_age) <= year <= current_year