"""
Utility Helper Functions
Common utility functions used across the application
"""

import time
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from config.api_keys import EXCHANGE_RATE_API_KEY

class RateLimiter:
    """Rate limiter to control API request frequency"""
    
    def __init__(self, requests_per_hour: int = 100):
        self.requests_per_hour = requests_per_hour
        self.requests_per_second = requests_per_hour / 3600
        self.last_request_time = 0
        self.request_times = []
        
        self.logger = logging.getLogger(__name__)
    
    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit"""
        current_time = time.time()
        
        # Remove requests older than 1 hour
        cutoff_time = current_time - 3600
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_hour:
            # Wait until the oldest request is over an hour old
            sleep_time = self.request_times[0] + 3600 - current_time
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        # Add minimum delay between requests
        min_delay = 1.0 / self.requests_per_second
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(time.time())
        self.last_request_time = time.time()

class CurrencyConverter:
    """Currency conversion utility using exchange rate API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = EXCHANGE_RATE_API_KEY
        self.base_url = "https://v6.exchangerate-api.com/v6"
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 hour
    
    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate between two currencies"""
        
        if from_currency == to_currency:
            return 1.0
        
        cache_key = f"{from_currency}_{to_currency}"
        current_time = time.time()
        
        # Check cache
        if (cache_key in self.cache and 
            cache_key in self.cache_expiry and 
            current_time < self.cache_expiry[cache_key]):
            return self.cache[cache_key]
        
        try:
            url = f"{self.base_url}/{self.api_key}/pair/{from_currency}/{to_currency}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('result') == 'success':
                rate = data.get('conversion_rate', 1.0)
                
                # Cache the result
                self.cache[cache_key] = rate
                self.cache_expiry[cache_key] = current_time + self.cache_duration
                
                return rate
            else:
                self.logger.error(f"Exchange rate API error: {data.get('error-type', 'Unknown error')}")
                return self.get_fallback_rate(from_currency, to_currency)
                
        except Exception as e:
            self.logger.error(f"Error getting exchange rate: {str(e)}")
            return self.get_fallback_rate(from_currency, to_currency)
    
    def get_fallback_rate(self, from_currency: str, to_currency: str) -> float:
        """Fallback exchange rates (approximate)"""
        
        fallback_rates = {
            'JPY_GBP': 0.0055,  # Approximate JPY to GBP
            'USD_GBP': 0.75,    # Approximate USD to GBP
            'EUR_GBP': 0.85     # Approximate EUR to GBP
        }
        
        rate_key = f"{from_currency}_{to_currency}"
        reverse_key = f"{to_currency}_{from_currency}"
        
        if rate_key in fallback_rates:
            return fallback_rates[rate_key]
        elif reverse_key in fallback_rates:
            return 1.0 / fallback_rates[reverse_key]
        else:
            self.logger.warning(f"No fallback rate available for {from_currency} to {to_currency}")
            return 1.0
    
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert amount from one currency to another"""
        
        if amount == 0:
            return 0.0
        
        rate = self.get_exchange_rate(from_currency, to_currency)
        return amount * rate

class DataValidator:
    """Data validation utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_vehicle_data(self, vehicle_data: Dict) -> bool:
        """Validate vehicle data structure and content"""
        
        required_fields = ['make', 'price']
        
        # Check required fields
        for field in required_fields:
            if field not in vehicle_data or vehicle_data[field] is None:
                return False
        
        # Validate price
        try:
            price = float(vehicle_data['price'])
            if price <= 0 or price > 1000000:  # Reasonable price range
                return False
        except (ValueError, TypeError):
            return False
        
        # Validate year if present
        if 'year' in vehicle_data and vehicle_data['year'] is not None:
            try:
                year = int(vehicle_data['year'])
                current_year = datetime.now().year
                if year < 1990 or year > current_year + 1:
                    return False
            except (ValueError, TypeError):
                return False
        
        # Validate mileage if present
        if 'mileage' in vehicle_data and vehicle_data['mileage'] is not None:
            try:
                mileage = float(vehicle_data['mileage'])
                if mileage < 0 or mileage > 500000:  # Reasonable mileage range
                    return False
            except (ValueError, TypeError):
                pass  # Mileage validation is not strict
        
        return True
    
    def validate_price_range(self, price: float, min_price: float = 500, max_price: float = 150000) -> bool:
        """Validate if price is within reasonable range"""
        return min_price <= price <= max_price
    
    def validate_year_range(self, year: int, max_age: int = 20) -> bool:
        """Validate if vehicle year is within acceptable range"""
        current_year = datetime.now().year
        return (current_year - max_age) <= year <= current_year
    
    def clean_text_field(self, text: str) -> str:
        """Clean and standardize text fields"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and standardize
        cleaned = " ".join(text.strip().split())
        
        # Remove special characters that might cause issues
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        return cleaned
    
    def standardize_make_name(self, make: str) -> str:
        """Standardize vehicle make names"""
        if not isinstance(make, str):
            return ""
        
        make = make.strip().title()
        
        # Common standardizations
        standardizations = {
            'Bmw': 'BMW',
            'Vw': 'Volkswagen',
            'Merc': 'Mercedes-Benz',
            'Mercedes': 'Mercedes-Benz',
            'Benz': 'Mercedes-Benz'
        }
        
        return standardizations.get(make, make)

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_times = {}
    
    def start_timer(self, operation_name: str):
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing and return duration"""
        if operation_name not in self.start_times:
            self.logger.warning(f"Timer {operation_name} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation_name]
        del self.start_times[operation_name]
        
        self.logger.info(f"{operation_name} completed in {duration:.2f} seconds")
        return duration

class APIResponseCache:
    """Simple cache for API responses"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.expiry_times = {}
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        current_time = time.time()
        
        if key in self.cache and key in self.expiry_times:
            if current_time < self.expiry_times[key]:
                self.logger.debug(f"Cache hit for key: {key}")
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.expiry_times[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value"""
        if ttl is None:
            ttl = self.default_ttl
        
        self.cache[key] = value
        self.expiry_times[key] = time.time() + ttl
        
        self.logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
    
    def clear(self) -> None:
        """Clear all cached values"""
        self.cache.clear()
        self.expiry_times.clear()
        self.logger.info("Cache cleared")
    
    def cleanup_expired(self) -> None:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.expiry_times.items()
            if current_time >= expiry_time
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.expiry_times[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

def format_currency(amount: float, currency: str = 'GBP') -> str:
    """Format currency amount for display"""
    
    symbols = {
        'GBP': '£',
        'USD': '$',
        'EUR': '€',
        'JPY': '¥'
    }
    
    symbol = symbols.get(currency, currency)
    
    if currency == 'JPY':
        # No decimal places for JPY
        return f"{symbol}{amount:,.0f}"
    else:
        return f"{symbol}{amount:,.2f}"

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format percentage for display"""
    return f"{value * 100:.{decimal_places}f}%"

def calculate_days_between(start_date: str, end_date: str = None) -> int:
    """Calculate days between two dates"""
    
    if end_date is None:
        end_date = datetime.now().isoformat()
    
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        return (end - start).days
        
    except Exception:
        return 0

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    
    if denominator == 0:
        return default
    
    return numerator / denominator

def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from text string"""
    
    if not isinstance(text, str):
        return None
    
    import re
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[£$€¥,]', '', text)
    
    # Extract number
    match = re.search(r'[\d,.]+', cleaned)
    
    if match:
        try:
            return float(match.group().replace(',', ''))
        except ValueError:
            pass
    
    return None

def batch_process(items: List[Any], batch_size: int = 100, delay: float = 0.1):
    """Process items in batches with delay"""
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
        
        if delay > 0 and i + batch_size < len(items):
            time.sleep(delay)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {str(e)}")
                        raise
            
        return wrapper
    return decorator