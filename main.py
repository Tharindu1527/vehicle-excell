# main.py - Pure Manual Import System (NO APIs)
#!/usr/bin/env python3
"""
Vehicle Import Analysis System - Pure Manual Import Mode
NO API dependencies - only manual CSV/Excel file imports
"""

import sys
import os
import logging
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collectors.manual_data_importer import ManualDataImporter
from src.analysis.profitability_analyzer import ProfitabilityAnalyzer
from src.analysis.scoring_engine import ScoringEngine
from src.database.database_manager import DatabaseManager
from src.dashboard.app import create_dashboard_app
from config.settings import Settings

class VehicleImportAnalyzer:
    """Pure manual import vehicle analyzer - no API dependencies"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.db_manager = DatabaseManager()
        
        # Initialize manual importer (NO API collectors)
        self.manual_importer = ManualDataImporter()
        
        # Initialize analyzers
        self.profitability_analyzer = ProfitabilityAnalyzer()
        self.scoring_engine = ScoringEngine()
        
        self.logger.info("Vehicle Import Analyzer initialized (Manual Import Only)")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def collect_all_data(self):
        """Data collection from manual imports only"""
        self.logger.info("Starting manual data collection")
        
        try:
            # Import UK data from CSV/Excel files
            self.logger.info("Importing UK market data from files...")
            uk_data = self.manual_importer.import_uk_data_batch()
            
            # Import Japan data from CSV/Excel files
            self.logger.info("Importing Japan auction data from files...")
            japan_data = self.manual_importer.import_japan_data_batch()
            
            # Store data in database
            if not uk_data.empty:
                self.db_manager.store_uk_data(uk_data)
                self.logger.info(f"Stored {len(uk_data)} UK records")
            else:
                self.logger.warning("No UK data found in import directory")
            
            if not japan_data.empty:
                self.db_manager.store_japan_data(japan_data)
                self.logger.info(f"Stored {len(japan_data)} Japan records")
            else:
                self.logger.warning("No Japan data found in import directory")
            
            if uk_data.empty and japan_data.empty:
                self.logger.warning("No data found in import directories. Please add CSV/Excel files to:")
                self.logger.warning(f"  UK data: {self.manual_importer.uk_import_dir}")
                self.logger.warning(f"  Japan data: {self.manual_importer.japan_import_dir}")
            
            self.logger.info("Manual data collection completed")
            
        except Exception as e:
            self.logger.error(f"Error during data collection: {str(e)}")
    
    def run_analysis(self):
        """Run profitability and scoring analysis"""
        self.logger.info("Starting analysis cycle")
        
        try:
            # Get latest data from database
            uk_data = self.db_manager.get_uk_data(days_back=30)
            japan_data = self.db_manager.get_japan_data(days_back=30)
            
            if uk_data.empty and japan_data.empty:
                self.logger.warning("No data available for analysis")
                self.logger.info("Please upload data using the dashboard or add files to import directories")
                return
            
            if uk_data.empty:
                self.logger.warning("No UK data available - upload UK vehicle data for analysis")
                return
            
            if japan_data.empty:
                self.logger.warning("No Japan data available - upload Japan auction data for analysis")
                return
            
            self.logger.info(f"Analyzing {len(uk_data)} UK records and {len(japan_data)} Japan records")
            
            # Calculate profitability
            profitability_results = self.profitability_analyzer.analyze(uk_data, japan_data)
            
            if profitability_results.empty:
                self.logger.warning("No profitable opportunities found")
                return
            
            # Generate scores
            scores = self.scoring_engine.calculate_scores(profitability_results)
            
            if not scores.empty:
                # Store analysis results
                self.db_manager.store_analysis_results(profitability_results, scores)
                
                self.logger.info(f"Analysis completed successfully - {len(scores)} opportunities analyzed")
                
                # Log top opportunities
                top_5 = scores.head(5)
                self.logger.info("Top 5 opportunities:")
                for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                    make = row.get('make', 'Unknown')
                    model = row.get('model', 'Vehicle')
                    score = row.get('final_score', 0)
                    margin = row.get('profit_margin', 0) * 100
                    profit = row.get('gross_profit', 0)
                    self.logger.info(f"  {idx}. {make} {model} - Score: {score:.1f}, Margin: {margin:.1f}%, Profit: £{profit:,.0f}")
            else:
                self.logger.warning("No scoring results generated")
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def run_manual_import_only(self):
        """Run manual import and analysis without dashboard"""
        self.logger.info("Running manual import and analysis...")
        
        # Initialize database
        self.db_manager.initialize_database()
        
        # Collect data from CSV/Excel files
        self.collect_all_data()
        
        # Run analysis if data exists
        self.run_analysis()
        
        self.logger.info("Manual import and analysis completed")
    
    def start_dashboard(self):
        """Start the web dashboard"""
        app = create_dashboard_app(self.db_manager)
        self.logger.info("Starting dashboard server on http://localhost:5000")
        self.logger.info("Dashboard features:")
        self.logger.info("  - Upload UK vehicle data (CSV/Excel)")
        self.logger.info("  - Upload Japan auction data (CSV/Excel)")
        self.logger.info("  - Run profitability analysis")
        self.logger.info("  - View opportunity rankings")
        self.logger.info("  - Portfolio optimization")
        self.logger.info("  - Export results to Excel")
        app.run(host='0.0.0.0', port=5000, debug=False)
    
    def run(self, dashboard_only=False, import_only=False):
        """Main application runner"""
        self.logger.info("Starting Vehicle Import Analyzer (Manual Import Mode)")
        
        # Initialize database
        self.db_manager.initialize_database()
        
        if import_only:
            # Just run import and analysis, then exit
            self.run_manual_import_only()
            return
        
        if not dashboard_only:
            # Run initial data collection and analysis if files exist
            self.collect_all_data()
            
            # Only run analysis if we have data
            uk_count = len(self.db_manager.get_uk_data(days_back=30))
            japan_count = len(self.db_manager.get_japan_data(days_back=30))
            
            if uk_count > 0 and japan_count > 0:
                self.run_analysis()
            else:
                self.logger.info("No initial data found - use dashboard to upload files")
        
        # Start dashboard
        self.start_dashboard()

def print_system_info():
    """Print system information and setup instructions"""
    print("🚗 VEHICLE IMPORT ANALYZER - MANUAL IMPORT MODE")
    print("=" * 60)
    print("📁 MODE: Pure Manual Import (No APIs)")
    print("   Data imported from CSV/Excel files only")
    print()
    print("📂 IMPORT DIRECTORIES:")
    print("   UK Data:    data_imports/uk_data/")
    print("   Japan Data: data_imports/japan_data/")
    print()
    print("📊 SUPPORTED FORMATS:")
    print("   - CSV files (.csv)")
    print("   - Excel files (.xlsx, .xls)")
    print()
    print("🌐 DASHBOARD FEATURES:")
    print("   - File upload interface")
    print("   - Drag & drop support")
    print("   - Real-time analysis")
    print("   - Interactive charts")
    print("   - Portfolio optimization")
    print("   - Excel export")
    print()

def check_setup():
    """Check if system is properly set up"""
    
    # Check if import directories exist
    uk_dir = "data_imports/uk_data"
    japan_dir = "data_imports/japan_data"
    
    uk_files = []
    japan_files = []
    
    if os.path.exists(uk_dir):
        uk_files = [f for f in os.listdir(uk_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
    if os.path.exists(japan_dir):
        japan_files = [f for f in os.listdir(japan_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    print("📋 SETUP STATUS:")
    print(f"   UK Directory:    {'✅ EXISTS' if os.path.exists(uk_dir) else '❌ MISSING'}")
    print(f"   Japan Directory: {'✅ EXISTS' if os.path.exists(japan_dir) else '❌ MISSING'}")
    print(f"   UK Files:        {len(uk_files)} files found")
    print(f"   Japan Files:     {len(japan_files)} files found")
    print()
    
    if not os.path.exists(uk_dir) or not os.path.exists(japan_dir):
        print("⚠️  SETUP REQUIRED:")
        print("   Run: python create_csv_templates.py")
        print("   This will create directories and sample data")
        print()
    
    if not uk_files and not japan_files:
        print("📁 NO DATA FILES FOUND:")
        print("   Use the dashboard upload feature, or")
        print("   Place CSV/Excel files in the import directories")
        print()
    
    return len(uk_files) > 0 or len(japan_files) > 0

def main():
    """Main function with enhanced setup and guidance"""
    
    parser = argparse.ArgumentParser(description='Vehicle Import Analyzer - Manual Import Mode')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Start dashboard only (skip initial data import)')
    parser.add_argument('--import-only', action='store_true',
                       help='Run import and analysis only (no dashboard)')
    parser.add_argument('--setup', action='store_true',
                       help='Show setup information and check status')
    
    args = parser.parse_args()
    
    print_system_info()
    
    if args.setup:
        check_setup()
        print("💡 NEXT STEPS:")
        print("   1. Create sample data: python create_csv_templates.py")
        print("   2. Start dashboard:    python main.py")
        print("   3. Upload your files:  Use dashboard upload buttons")
        print("   4. Run analysis:       Click 'Run Analysis' button")
        return
    
    has_data = check_setup()
    
    if args.dashboard_only:
        print("📊 Starting dashboard only...")
    elif args.import_only:
        print("📥 Running import and analysis only...")
        if not has_data:
            print("❌ No data files found. Add files to import directories first.")
            return
    else:
        print("🚀 Starting full system...")
    
    try:
        analyzer = VehicleImportAnalyzer()
        analyzer.run(
            dashboard_only=args.dashboard_only,
            import_only=args.import_only
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Check if all required packages are installed: pip install -r requirements.txt")
        print("   2. Ensure import directories exist: python create_csv_templates.py")
        print("   3. Check system status: python main.py --setup")
        print("   4. View logs in: logs/system.log")
        sys.exit(1)

if __name__ == "__main__":
    main()