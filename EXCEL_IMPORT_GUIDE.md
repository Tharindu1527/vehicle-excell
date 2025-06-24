
# Enhanced Vehicle Import Analyzer Configuration

## New Features Added:
1. Manual Excel/CSV data import
2. Enhanced dashboard with upload interface  
3. Template downloads for proper data format
4. Upload history tracking
5. Data management tools

## File Upload Locations:
- UK Market Data: /upload-uk-data
- Japan Auction Data: /upload-japan-data
- Data Import Page: /data-import

## Expected Excel Columns:

### UK Market Data:
- Make (required): Vehicle manufacturer
- Model (required): Vehicle model  
- Year (required): Year of manufacture
- Price (required): Selling price in GBP
- Mileage: Vehicle mileage
- Fuel_Type: Petrol/Diesel/Hybrid/Electric
- Transmission: Manual/Automatic
- Location: UK location/region

### Japan Auction Data:
- Make (required): Vehicle manufacturer
- Model (required): Vehicle model
- Year (required): Year of manufacture  
- Final_Price_JPY: Auction price in Japanese Yen
- Final_Price_GBP: Auction price converted to GBP
- Mileage_KM: Vehicle mileage in kilometers
- Grade: Auction grade (A, B, C, etc.)
- Auction_House: Name of auction house
- Total_Landed_Cost: Total cost including import fees
- Fuel_Type: Petrol/Diesel/Hybrid/Electric

## Usage Instructions:
1. Start system: python enhanced_main.py
2. Open browser: http://localhost:5000
3. Click "Import Data" button
4. Download templates or upload your own Excel files
5. Upload UK market data and Japan auction data
6. Click "Run Analysis" to process data
7. View results on main dashboard

## File Format Support:
- Excel (.xlsx, .xls)
- CSV (.csv)
- Maximum file size: 16MB

## Data Validation:
- Automatic column mapping for similar names
- Price validation and currency conversion
- Data type conversion and cleaning
- Duplicate removal
- Error reporting and logging
