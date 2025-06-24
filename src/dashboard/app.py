# src/dashboard/app.py
"""
Updated Web Dashboard Application with File Upload Functionality
Pure manual import system - NO API dependencies
"""

from flask import Flask, render_template, jsonify, request, send_file, flash, redirect, url_for
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

from src.database.database_manager import DatabaseManager
from src.analysis.scoring_engine import ScoringEngine
from src.data_collectors.manual_data_importer import ManualDataImporter
from src.analysis.profitability_analyzer import ProfitabilityAnalyzer
from config.settings import Settings

def create_dashboard_app(db_manager: DatabaseManager) -> Flask:
    """Create and configure the Flask dashboard application with file upload"""
    
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
    manual_importer = ManualDataImporter()
    profitability_analyzer = ProfitabilityAnalyzer()
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def dashboard():
        """Main dashboard page with upload functionality"""
        try:
            logger.info("Dashboard page requested")
            return render_template('dashboard.html')
        except Exception as e:
            logger.error(f"Error loading dashboard template: {str(e)}")
            return f"Error loading dashboard: {str(e)}", 500
    
    @app.route('/upload')
    def upload_page():
        """File upload page"""
        try:
            return render_template('upload.html')
        except Exception as e:
            logger.error(f"Error loading upload template: {str(e)}")
            return f"Error loading upload page: {str(e)}", 500
    
    @app.route('/api/upload/<data_type>', methods=['POST'])
    def upload_file(data_type):
        """Handle file uploads for UK or Japan data"""
        
        logger.info(f"Upload endpoint called for data_type: {data_type}")
        
        if data_type not in ['uk', 'japan']:
            logger.error(f"Invalid data type: {data_type}")
            return jsonify({'error': 'Invalid data type'}), 400
        
        try:
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed. Use CSV or Excel files.'}), 400
            
            # Save and process the file
            file_path = manual_importer.save_uploaded_file(file, data_type)
            result = manual_importer.process_uploaded_file(file_path, data_type)
            
            if result['success']:
                # Store in database
                if data_type == 'uk':
                    uk_data = manual_importer.import_file(result['file_path'], 'uk')
                    if not uk_data.empty:
                        db_manager.store_uk_data(uk_data)
                        logger.info(f"Stored {len(uk_data)} UK records from upload")
                else:
                    japan_data = manual_importer.import_file(result['file_path'], 'japan')
                    if not japan_data.empty:
                        db_manager.store_japan_data(japan_data)
                        logger.info(f"Stored {len(japan_data)} Japan records from upload")
                
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'count': result['count'],
                    'data_type': data_type
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result['message']
                }), 400
                
        except Exception as e:
            logger.error(f"Error uploading {data_type} file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Upload failed: {str(e)}'
            }), 500
    
    @app.route('/api/run-analysis', methods=['POST'])
    def run_analysis():
        """Run analysis on uploaded data"""
        
        try:
            logger.info("Running analysis on uploaded data")
            
            # Get data from database
            uk_data = db_manager.get_uk_data(days_back=30)
            japan_data = db_manager.get_japan_data(days_back=30)
            
            if uk_data.empty and japan_data.empty:
                return jsonify({
                    'success': False,
                    'error': 'No data available for analysis. Please upload UK and Japan data first.'
                }), 400
            
            # Run profitability analysis
            profitability_results = profitability_analyzer.analyze(uk_data, japan_data)
            
            if profitability_results.empty:
                return jsonify({
                    'success': False,
                    'error': 'No profitable opportunities found in the data.'
                }), 400
            
            # Generate scores
            scores = scoring_engine.calculate_scores(profitability_results)
            
            if scores.empty:
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate opportunity scores.'
                }), 400
            
            # Store analysis results
            db_manager.store_analysis_results(profitability_results, scores)
            
            # Get top opportunities for response
            top_5 = scores.head(5)
            opportunities = []
            
            for _, row in top_5.iterrows():
                opportunities.append({
                    'make': str(row.get('make', 'Unknown')),
                    'model': str(row.get('model', 'Vehicle')),
                    'year': int(row.get('year', 2018)),
                    'score': float(row.get('final_score', 0)),
                    'profit_margin': float(row.get('profit_margin', 0)),
                    'expected_profit': float(row.get('gross_profit', 0))
                })
            
            return jsonify({
                'success': True,
                'message': f'Analysis completed successfully. Found {len(scores)} opportunities.',
                'total_opportunities': len(scores),
                'top_opportunities': opportunities,
                'uk_records': len(uk_data),
                'japan_records': len(japan_data)
            })
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }), 500
    
    @app.route('/api/data-status')
    def data_status():
        """Get current data status"""
        try:
            uk_count = len(db_manager.get_uk_data(days_back=30))
            japan_count = len(db_manager.get_japan_data(days_back=30))
            analysis_count = len(db_manager.get_latest_analysis_results(limit=10))
            
            return jsonify({
                'uk_records': uk_count,
                'japan_records': japan_count,
                'analysis_results': analysis_count,
                'has_uk_data': uk_count > 0,
                'has_japan_data': japan_count > 0,
                'has_analysis': analysis_count > 0,
                'can_analyze': uk_count > 0 and japan_count > 0
            })
            
        except Exception as e:
            logger.error(f"Error getting data status: {str(e)}")
            return jsonify({
                'error': str(e),
                'uk_records': 0,
                'japan_records': 0,
                'analysis_results': 0
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
                'mode': 'manual_import_only',
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
            
            # Test database connectivity first
            if not os.path.exists(db_manager.db_path):
                logger.error(f"Database file does not exist: {db_manager.db_path}")
                return jsonify({
                    'error': 'Database not found',
                    'message': 'Upload data files to initialize the system'
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
                'last_updated': datetime.now().isoformat(),
                'data_mode': 'manual_import'
            }
            
            logger.info(f"Summary prepared with {len(top_opportunities)} opportunities")
            return jsonify(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'message': 'Please upload data files and run analysis'
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
            logger.error(f"Traceback: {traceback.format_exc()}")
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
            # Return empty trends data instead of error
            return jsonify({
                'uk_price_by_make': {'x': [], 'y': []},
                'japan_price_by_make': {'x': [], 'y': []},
                'message': 'No trend data available'
            })
    
    @app.route('/api/export/excel')
    def api_export_excel():
        """Export analysis results to Excel"""
        try:
            # Get all analysis results
            analysis_results = db_manager.get_latest_analysis_results(limit=1000)
            
            if analysis_results.empty:
                return jsonify({'error': 'No data to export'}), 400
            
            # Create Excel file in memory
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main analysis sheet
                analysis_results.to_excel(writer, sheet_name='Analysis Results', index=False)
                
                # Summary sheet
                summary_data = generate_export_summary(analysis_results)
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Top opportunities
                top_20 = analysis_results.head(20)
                top_20.to_excel(writer, sheet_name='Top 20 Opportunities', index=False)
            
            output.seek(0)
            
            # Generate filename with timestamp
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
    
    @app.route('/status')
    def status_page():
        """Simple status page for debugging"""
        try:
            stats = db_manager.get_database_stats()
            uk_count = len(db_manager.get_uk_data(days_back=7))
            japan_count = len(db_manager.get_japan_data(days_back=7))
            analysis_count = len(db_manager.get_latest_analysis_results(limit=10))
            
            status_html = f"""
            <html>
            <head><title>System Status - Manual Import Mode</title></head>
            <body>
                <h1>Vehicle Import Analyzer - System Status</h1>
                <h2>🎯 Mode: Manual Import Only (No APIs)</h2>
                
                <h2>Database Status</h2>
                <p>Database file: {db_manager.db_path}</p>
                <p>Database exists: {os.path.exists(db_manager.db_path)}</p>
                <p>Database stats: {stats}</p>
                
                <h2>Data Counts (Last 7 days)</h2>
                <p>UK Records: {uk_count}</p>
                <p>Japan Records: {japan_count}</p>
                <p>Analysis Results: {analysis_count}</p>
                
                <h2>Upload Directories</h2>
                <p>UK Data: data_imports/uk_data/</p>
                <p>Japan Data: data_imports/japan_data/</p>
                
                <h2>Navigation</h2>
                <p><a href="/upload">📁 Upload Files</a></p>
                <p><a href="/">📊 Main Dashboard</a></p>
                <p><a href="/api/test">🔧 Test API</a></p>
                <p><a href="/api/summary">📋 Summary API</a></p>
                <p><a href="/api/data-status">📈 Data Status</a></p>
                
                <h2>Actions</h2>
                <p>To add data: Use the upload page or place files in data_imports/ directories</p>
                <p>Supported formats: CSV, Excel (.xlsx, .xls)</p>
                
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
                # Price trends by make
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
                # Japan auction prices
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
                'Data Source': 'Manual Import',
                'Top Make by Count': analysis_results['make'].value_counts().index[0] if not analysis_results.empty else 'N/A',
                'Average Final Score': f"{analysis_results['final_score'].mean():.1f}" if 'final_score' in analysis_results.columns else 'N/A'
            }
        except Exception as e:
            logger.error(f"Error generating export summary: {str(e)}")
            return {'error': 'Failed to generate summary'}
    
    return app