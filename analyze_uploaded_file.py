# analyze_uploaded_file.py
"""
Analyze uploaded files to understand why they're failing
"""

import pandas as pd
import os
import sys

def analyze_file(file_path):
    """Analyze a file to understand its structure"""
    
    print(f"🔍 ANALYZING FILE: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # File info
    file_size = os.path.getsize(file_path)
    print(f"📁 File size: {file_size:,} bytes")
    print(f"📄 File extension: {os.path.splitext(file_path)[1]}")
    
    try:
        # Try different reading methods
        print("\n📖 READING ATTEMPTS:")
        print("-" * 30)
        
        # Method 1: Standard pandas read
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            print(f"✅ Standard read successful")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Column names: {list(df.columns)}")
            
            if len(df) > 0:
                print(f"\n📋 FIRST FEW ROWS:")
                print(df.head(3).to_string())
                
                print(f"\n📊 DATA TYPES:")
                for col in df.columns:
                    print(f"   {col}: {df[col].dtype}")
                
                print(f"\n🔍 NON-NULL COUNTS:")
                for col in df.columns:
                    non_null = df[col].notna().sum()
                    print(f"   {col}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
                
                return df
            else:
                print("⚠️  File is empty")
                
        except Exception as e:
            print(f"❌ Standard read failed: {e}")
        
        # Method 2: Try different encodings for CSV
        if file_path.endswith('.csv'):
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    print(f"\n🔤 Trying encoding: {encoding}")
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ Success with {encoding}")
                    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                    return df
                except Exception as e:
                    print(f"❌ Failed with {encoding}: {e}")
        
        # Method 3: Try different separators
        if file_path.endswith('.csv'):
            separators = [',', ';', '\t', '|']
            for sep in separators:
                try:
                    print(f"\n📊 Trying separator: '{sep}'")
                    df = pd.read_csv(file_path, sep=sep)
                    if len(df.columns) > 1:  # More than one column suggests correct separator
                        print(f"✅ Success with separator '{sep}'")
                        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                        return df
                except Exception as e:
                    print(f"❌ Failed with separator '{sep}': {e}")
        
        # Method 4: Read as text to see raw content
        print(f"\n📝 RAW FILE CONTENT (first 500 chars):")
        print("-" * 40)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)
                print(content)
                print("\n...")
        except Exception as e:
            print(f"❌ Could not read as text: {e}")
        
        # Method 5: Check if it's actually an Excel file
        if file_path.endswith(('.xlsx', '.xls')):
            try:
                print(f"\n📊 EXCEL ANALYSIS:")
                print("-" * 20)
                
                # Try different Excel engines
                engines = ['openpyxl', 'xlrd'] if file_path.endswith('.xls') else ['openpyxl']
                
                for engine in engines:
                    try:
                        print(f"🔧 Trying engine: {engine}")
                        df = pd.read_excel(file_path, engine=engine)
                        print(f"✅ Success with {engine}")
                        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                        return df
                    except Exception as e:
                        print(f"❌ Failed with {engine}: {e}")
                
                # Try reading all sheets
                try:
                    print(f"\n📋 CHECKING ALL SHEETS:")
                    xl_file = pd.ExcelFile(file_path)
                    print(f"   Sheet names: {xl_file.sheet_names}")
                    
                    for sheet in xl_file.sheet_names:
                        try:
                            df = pd.read_excel(file_path, sheet_name=sheet)
                            print(f"   Sheet '{sheet}': {len(df)} rows, {len(df.columns)} columns")
                            if len(df) > 0:
                                return df
                        except Exception as e:
                            print(f"   Sheet '{sheet}': Failed - {e}")
                            
                except Exception as e:
                    print(f"❌ Could not analyze sheets: {e}")
            
            except Exception as e:
                print(f"❌ Excel analysis failed: {e}")
        
        print(f"\n❌ ALL READING METHODS FAILED")
        return None
        
    except Exception as e:
        print(f"💥 FATAL ERROR: {e}")
        return None

def suggest_fixes(file_path, df=None):
    """Suggest fixes based on analysis"""
    
    print(f"\n💡 SUGGESTIONS:")
    print("-" * 20)
    
    if df is None:
        print("🔧 FILE READING ISSUES:")
        print("   1. Check if file is corrupted")
        print("   2. Try saving as CSV with UTF-8 encoding")
        print("   3. Ensure file has proper headers")
        print("   4. Check for special characters in data")
        print("   5. Try opening in Excel/LibreOffice first")
        
    elif len(df) == 0:
        print("📝 EMPTY FILE:")
        print("   1. File has headers but no data rows")
        print("   2. Add some sample data to test")
        
    else:
        # Check for minimum required columns
        columns = [col.lower().strip() for col in df.columns]
        
        has_make = any('make' in col or 'brand' in col for col in columns)
        has_price = any('price' in col or 'cost' in col or 'value' in col for col in columns)
        has_year = any('year' in col for col in columns)
        
        print("📊 DATA QUALITY CHECKS:")
        print(f"   Has make/brand column: {'✅' if has_make else '❌'}")
        print(f"   Has price/cost column: {'✅' if has_price else '❌'}")
        print(f"   Has year column: {'✅' if has_year else '❌'}")
        
        if not has_make:
            print("\n🔧 MISSING MAKE/BRAND:")
            print("   Add a column with vehicle manufacturer (Toyota, Honda, etc.)")
            
        if not has_price:
            print("\n🔧 MISSING PRICE:")
            print("   Add a column with vehicle price (numeric values)")
            
        # Check for data in important columns
        if has_make:
            make_col = next((col for col in df.columns if 'make' in col.lower() or 'brand' in col.lower()), None)
            if make_col:
                non_null_makes = df[make_col].notna().sum()
                print(f"\n📋 MAKE/BRAND DATA: {non_null_makes}/{len(df)} rows have data")
                if non_null_makes > 0:
                    unique_makes = df[make_col].value_counts().head(5)
                    print(f"   Top makes: {list(unique_makes.index)}")
        
        if has_price:
            price_col = next((col for col in df.columns if 'price' in col.lower() or 'cost' in col.lower()), None)
            if price_col:
                non_null_prices = df[price_col].notna().sum()
                numeric_prices = pd.to_numeric(df[price_col], errors='coerce').notna().sum()
                print(f"\n💰 PRICE DATA: {non_null_prices}/{len(df)} rows have data")
                print(f"   Numeric prices: {numeric_prices}/{len(df)} rows")
                if numeric_prices > 0:
                    prices = pd.to_numeric(df[price_col], errors='coerce')
                    print(f"   Price range: £{prices.min():,.0f} - £{prices.max():,.0f}")

def analyze_recent_uploads():
    """Analyze recently uploaded files"""
    
    upload_dirs = ['data_imports/uploads', 'uploads/uk', 'uploads/japan']
    
    print("🔍 ANALYZING RECENT UPLOADS")
    print("=" * 40)
    
    files_found = []
    
    for upload_dir in upload_dirs:
        if os.path.exists(upload_dir):
            files = [f for f in os.listdir(upload_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
            for file in files:
                file_path = os.path.join(upload_dir, file)
                files_found.append(file_path)
    
    if not files_found:
        print("❌ No uploaded files found in:")
        for dir in upload_dirs:
            print(f"   {dir}")
        print("\n💡 Try uploading a file first")
        return
    
    print(f"📁 Found {len(files_found)} files:")
    for file_path in files_found:
        print(f"   {file_path}")
    
    # Analyze the most recent file
    latest_file = max(files_found, key=os.path.getmtime)
    print(f"\n🔍 Analyzing most recent: {latest_file}")
    
    df = analyze_file(latest_file)
    suggest_fixes(latest_file, df)

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        df = analyze_file(file_path)
        suggest_fixes(file_path, df)
    else:
        analyze_recent_uploads()

if __name__ == "__main__":
    main()