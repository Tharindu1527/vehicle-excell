# create_csv_templates.py
"""
Create sample CSV templates for manual data import
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import random

def create_uk_sample_csv():
    """Create sample UK vehicle CSV template"""
    
    # Sample UK vehicle data
    uk_sample_data = [
        {
            'make': 'Toyota',
            'model': 'Prius',
            'year': 2020,
            'price': 18500,
            'mileage': 32000,
            'fuel_type': 'Hybrid',
            'transmission': 'Automatic',
            'body_type': 'Hatchback',
            'condition': 'Used',
            'title': '2020 Toyota Prius 1.8 VVT-i Hybrid Business Edition Plus',
            'url': 'https://example.com/listing1',
            'location': 'London',
            'seller': 'Premium Motors',
            'currency': 'GBP'
        },
        {
            'make': 'Honda',
            'model': 'Civic',
            'year': 2019,
            'price': 16200,
            'mileage': 28500,
            'fuel_type': 'Petrol',
            'transmission': 'Manual',
            'body_type': 'Hatchback',
            'condition': 'Used',
            'title': '2019 Honda Civic 1.0 VTEC Turbo EX',
            'url': 'https://example.com/listing2',
            'location': 'Manchester',
            'seller': 'City Motors',
            'currency': 'GBP'
        },
        {
            'make': 'Nissan',
            'model': 'Qashqai',
            'year': 2021,
            'price': 22800,
            'mileage': 15000,
            'fuel_type': 'Petrol',
            'transmission': 'Automatic',
            'body_type': 'SUV',
            'condition': 'Used',
            'title': '2021 Nissan Qashqai 1.3 DiG-T Tekna',
            'url': 'https://example.com/listing3',
            'location': 'Birmingham',
            'seller': 'Midlands Motors',
            'currency': 'GBP'
        },
        {
            'make': 'Mazda',
            'model': 'CX-5',
            'year': 2018,
            'price': 19500,
            'mileage': 42000,
            'fuel_type': 'Diesel',
            'transmission': 'Manual',
            'body_type': 'SUV',
            'condition': 'Used',
            'title': '2018 Mazda CX-5 2.2d Sport Nav',
            'url': 'https://example.com/listing4',
            'location': 'Leeds',
            'seller': 'Yorkshire Motors',
            'currency': 'GBP'
        },
        {
            'make': 'Subaru',
            'model': 'Outback',
            'year': 2020,
            'price': 26500,
            'mileage': 25000,
            'fuel_type': 'Petrol',
            'transmission': 'Automatic',
            'body_type': 'Estate',
            'condition': 'Used',
            'title': '2020 Subaru Outback 2.5i SE Premium',
            'url': 'https://example.com/listing5',
            'location': 'Edinburgh',
            'seller': 'Scottish Motors',
            'currency': 'GBP'
        }
    ]
    
    # Create DataFrame and save
    df = pd.DataFrame(uk_sample_data)
    
    # Create directory
    os.makedirs('data_imports/uk_data', exist_ok=True)
    
    # Save template and sample
    df.to_csv('data_imports/uk_data/uk_sample_data.csv', index=False)
    
    # Create empty template
    template_df = pd.DataFrame(columns=df.columns)
    template_df.to_csv('data_imports/uk_data/uk_template.csv', index=False)
    
    print("✅ Created UK CSV files:")
    print("   📁 data_imports/uk_data/uk_sample_data.csv (5 sample records)")
    print("   📁 data_imports/uk_data/uk_template.csv (empty template)")

def create_japan_sample_csv():
    """Create sample Japan auction CSV template"""
    
    # Sample Japan auction data
    japan_sample_data = [
        {
            'make': 'Toyota',
            'model': 'Prius',
            'year': 2019,
            'final_price_jpy': 1800000,
            'final_price_gbp': 9900,
            'mileage_km': 45000,
            'mileage_miles': 27961,
            'auction_house': 'USS Tokyo',
            'auction_date': '2024-01-15',
            'grade': 'A',
            'fuel_type': 'Hybrid',
            'transmission': 'CVT',
            'body_type': 'Hatchback',
            'colour': 'White',
            'title': '2019 Toyota Prius Hybrid Grade A',
            'total_landed_cost': 13365
        },
        {
            'make': 'Honda',
            'model': 'Fit',
            'year': 2020,
            'final_price_jpy': 1200000,
            'final_price_gbp': 6600,
            'mileage_km': 32000,
            'mileage_miles': 19883,
            'auction_house': 'USS Osaka',
            'auction_date': '2024-01-20',
            'grade': 'B',
            'fuel_type': 'Hybrid',
            'transmission': 'CVT',
            'body_type': 'Hatchback',
            'colour': 'Silver',
            'title': '2020 Honda Fit Hybrid Grade B',
            'total_landed_cost': 8910
        },
        {
            'make': 'Nissan',
            'model': 'Note',
            'year': 2018,
            'final_price_jpy': 1500000,
            'final_price_gbp': 8250,
            'mileage_km': 58000,
            'mileage_miles': 36039,
            'auction_house': 'TAA Kansai',
            'auction_date': '2024-01-25',
            'grade': 'B',
            'fuel_type': 'Hybrid',
            'transmission': 'CVT',
            'body_type': 'Hatchback',
            'colour': 'Blue',
            'title': '2018 Nissan Note e-Power Grade B',
            'total_landed_cost': 11138
        },
        {
            'make': 'Mazda',
            'model': 'CX-5',
            'year': 2017,
            'final_price_jpy': 2200000,
            'final_price_gbp': 12100,
            'mileage_km': 65000,
            'mileage_miles': 40389,
            'auction_house': 'USS Saitama',
            'auction_date': '2024-02-01',
            'grade': 'C',
            'fuel_type': 'Petrol',
            'transmission': 'AT',
            'body_type': 'SUV',
            'colour': 'Red',
            'title': '2017 Mazda CX-5 Grade C',
            'total_landed_cost': 16335
        },
        {
            'make': 'Subaru',
            'model': 'Forester',
            'year': 2019,
            'final_price_jpy': 2800000,
            'final_price_gbp': 15400,
            'mileage_km': 38000,
            'mileage_miles': 23612,
            'auction_house': 'USS Yokohama',
            'auction_date': '2024-02-05',
            'grade': 'A',
            'fuel_type': 'Petrol',
            'transmission': 'CVT',
            'body_type': 'SUV',
            'colour': 'Black',
            'title': '2019 Subaru Forester Grade A',
            'total_landed_cost': 20790
        },
        {
            'make': 'Lexus',
            'model': 'IS',
            'year': 2020,
            'final_price_jpy': 4500000,
            'final_price_gbp': 24750,
            'mileage_km': 25000,
            'mileage_miles': 15534,
            'auction_house': 'USS Tokyo',
            'auction_date': '2024-02-10',
            'grade': 'S',
            'fuel_type': 'Hybrid',
            'transmission': 'CVT',
            'body_type': 'Sedan',
            'colour': 'Pearl White',
            'title': '2020 Lexus IS 300h Grade S',
            'total_landed_cost': 33413
        }
    ]
    
    # Create DataFrame and save
    df = pd.DataFrame(japan_sample_data)
    
    # Create directory
    os.makedirs('data_imports/japan_data', exist_ok=True)
    
    # Save template and sample
    df.to_csv('data_imports/japan_data/japan_sample_data.csv', index=False)
    
    # Create empty template
    template_df = pd.DataFrame(columns=df.columns)
    template_df.to_csv('data_imports/japan_data/japan_template.csv', index=False)
    
    print("✅ Created Japan CSV files:")
    print("   📁 data_imports/japan_data/japan_sample_data.csv (6 sample records)")
    print("   📁 data_imports/japan_data/japan_template.csv (empty template)")

def create_readme():
    """Create README file for data imports"""
    
    readme_content = """# Vehicle Import Data - Manual Import Guide

## Overview
This directory contains CSV files for manual data import into the Vehicle Import Analyzer system.

## Directory Structure
```
data_imports/
├── uk_data/           # UK vehicle market data
│   ├── uk_template.csv       # Empty template
│   └── uk_sample_data.csv    # Sample data
└── japan_data/        # Japan auction data
    ├── japan_template.csv    # Empty template
    └── japan_sample_data.csv # Sample data
```

## How to Import Your Data

1. **Prepare your CSV files** using the templates as guides
2. **Place files** in the appropriate directories:
   - UK data: `data_imports/uk_data/`
   - Japan data: `data_imports/japan_data/`
3. **Run the import** using: `python manual_import_runner.py`

## UK Data CSV Format

### Required Columns
- `make` - Vehicle manufacturer (Toyota, Honda, etc.)
- `price` - Price in GBP

### Recommended Columns
- `model` - Vehicle model
- `year` - Model year
- `mileage` - Vehicle mileage
- `fuel_type` - Petrol, Diesel, Hybrid, Electric
- `transmission` - Automatic, Manual
- `title` - Vehicle description
- `location` - Vehicle location
- `url` - Listing URL

### Example
```csv
make,model,year,price,mileage,fuel_type,transmission
Toyota,Prius,2020,18500,32000,Hybrid,Automatic
Honda,Civic,2019,16200,28500,Petrol,Manual
```

## Japan Data CSV Format

### Required Columns
- `make` - Vehicle manufacturer

### Recommended Columns
- `model` - Vehicle model
- `year` - Model year
- `final_price_jpy` - Auction price in Japanese Yen
- `final_price_gbp` - Auction price in British Pounds
- `mileage_km` - Mileage in kilometers
- `auction_house` - Auction location
- `grade` - Auction grade (S, A, B, C)
- `fuel_type` - Fuel type
- `transmission` - Transmission type

### Example
```csv
make,model,year,final_price_jpy,final_price_gbp,mileage_km,auction_house,grade
Toyota,Prius,2019,1800000,9900,45000,USS Tokyo,A
Honda,Fit,2020,1200000,6600,32000,USS Osaka,B
```

## Import Commands

- `python manual_import_runner.py` - Interactive import
- `python manual_import_runner.py validate` - Validate files only
- `python manual_import_runner.py full` - Complete import and analysis
- `python manual_import_runner.py format` - Show detailed format guide

## Tips

1. **Column names are flexible** - the system will try to match similar names
2. **Multiple files supported** - you can have multiple CSV files in each directory
3. **Missing data handled** - missing values will be filled with reasonable defaults
4. **Automatic validation** - invalid records are filtered out automatically
5. **Currency conversion** - JPY prices will be converted to GBP automatically

## Data Sources

You can obtain vehicle data from:
- **UK**: AutoTrader, Motors.co.uk, CarGurus, eBay Motors
- **Japan**: Yahoo Auctions Japan, BH Auction, auction house websites

Export search results as CSV files and place them in the appropriate directories.

## Troubleshooting

- **No data imported**: Check column names match the expected format
- **Analysis fails**: Ensure you have both UK and Japan data
- **Price issues**: Verify prices are in the correct currency (GBP for UK, JPY for Japan)

For detailed help, run: `python manual_import_runner.py format`
"""
    
    with open('data_imports/README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ Created README.md with detailed instructions")

def main():
    """Create all templates and setup files"""
    print("🔧 CREATING CSV TEMPLATES AND SETUP")
    print("=" * 50)
    
    # Create directories
    os.makedirs('data_imports', exist_ok=True)
    
    # Create templates
    create_uk_sample_csv()
    create_japan_sample_csv()
    create_readme()
    
    print("\n🎉 SETUP COMPLETE!")
    print("\n📖 Next steps:")
    print("1. Check the data_imports/README.md for detailed instructions")
    print("2. Add your CSV files to data_imports/uk_data/ and data_imports/japan_data/")
    print("3. Run: python manual_import_runner.py")
    
    print("\n📊 Sample data included:")
    print("- UK: 5 sample vehicle records")
    print("- Japan: 6 sample auction records")
    print("- You can test the system with this sample data immediately!")

if __name__ == "__main__":
    main()