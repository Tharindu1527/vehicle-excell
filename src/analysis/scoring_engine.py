"""
Vehicle Scoring Engine with Portfolio Optimization
Advanced scoring system for ranking vehicle import opportunities
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler

class ScoringEngine:
    """Advanced scoring engine for vehicle import opportunities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        
        # Scoring weights (adjustable based on business priorities)
        self.weights = {
            'profitability': 0.35,  # Most important factor
            'market_demand': 0.25,
            'risk_assessment': 0.20,
            'liquidity': 0.15,      # How fast it sells
            'market_trends': 0.05
        }
        
        self.logger.info("Scoring engine initialized")
    
    def calculate_scores(self, profitability_data: pd.DataFrame) -> pd.DataFrame:
        """Main scoring calculation"""
        self.logger.info("Calculating vehicle scores")
        
        if profitability_data.empty:
            return pd.DataFrame()
        
        try:
            # Calculate individual component scores
            scores = profitability_data.copy()
            
            # Profitability score
            scores['profitability_score'] = self.calculate_profitability_score(scores)
            
            # Market demand score
            scores['market_demand_score'] = self.calculate_market_demand_score(scores)
            
            # Risk assessment score
            scores['risk_assessment_score'] = self.calculate_risk_assessment_score(scores)
            
            # Liquidity score (speed of sale)
            scores['liquidity_score'] = self.calculate_liquidity_score(scores)
            
            # Market trends score
            scores['market_trends_score'] = self.calculate_market_trends_score(scores)
            
            # Calculate weighted final score
            scores['final_score'] = self.calculate_weighted_score(scores)
            
            # Add percentile rankings
            scores = self.add_percentile_rankings(scores)
            
            # Add score interpretations
            scores['score_grade'] = scores['final_score'].apply(self.get_score_grade)
            scores['investment_category'] = scores['final_score'].apply(self.get_investment_category)
            
            self.logger.info(f"Scoring completed for {len(scores)} vehicles")
            
            return scores.sort_values('final_score', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error in scoring calculation: {str(e)}")
            return profitability_data
    
    def calculate_profitability_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate profitability score (0-100)"""
        
        # Multiple profitability metrics
        profit_margin_score = np.clip(data['profit_margin'] * 100, 0, 50)
        roi_score = np.clip(data['roi'] * 50, 0, 30)
        absolute_profit_score = np.clip(data['gross_profit'] / 1000, 0, 20)
        
        return profit_margin_score + roi_score + absolute_profit_score
    
    def calculate_market_demand_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market demand score (0-100)"""
        
        # Demand indicators
        demand_base = data['demand_score'] * 10  # 0-100
        
        # Market share bonus
        market_share_bonus = np.clip(data['market_share'] * 10000, 0, 20)
        
        # Listing density bonus
        listing_bonus = np.clip(data['listings_density'] / 2, 0, 15)
        
        # Sample size confidence
        sample_confidence = np.clip(data['uk_sample_size'] / 5, 0, 15)
        
        total_score = demand_base + market_share_bonus + listing_bonus + sample_confidence
        
        return np.clip(total_score, 0, 100)
    
    def calculate_risk_assessment_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate risk assessment score (0-100, higher is better/lower risk)"""
        
        # Invert risk score (lower risk = higher score)
        base_risk_score = (10 - data['risk_score']) * 10
        
        # Price stability bonus
        price_volatility = data.get('price_volatility_uk', 0.2)
        stability_bonus = np.clip((0.3 - price_volatility) * 100, 0, 20)
        
        # Sample size confidence (more data = less risk)
        sample_confidence = np.clip(data['uk_sample_size'] / 3, 0, 15)
        
        # Grade quality bonus (for Japan auction data)
        grade_bonus = np.clip((data.get('japan_avg_grade', 5) - 5) * 5, 0, 10)
        
        total_score = base_risk_score + stability_bonus + sample_confidence + grade_bonus
        
        return np.clip(total_score, 0, 100)
    
    def calculate_liquidity_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score (how fast vehicles sell) (0-100)"""
        
        # Estimate based on average days listed (inverted)
        avg_days = data.get('avg_days_listed', 30)
        
        # Convert to score (faster sale = higher score)
        days_score = np.clip(100 - (avg_days - 7) * 2, 10, 100)
        
        # Price range adjustment (mid-range cars often sell faster)
        price_range_score = self.get_price_range_liquidity_score(data['uk_avg_price'])
        
        # Market demand adjustment
        demand_adjustment = data['demand_score'] * 5
        
        total_score = days_score * 0.6 + price_range_score * 0.3 + demand_adjustment * 0.1
        
        return np.clip(total_score, 0, 100)
    
    def get_price_range_liquidity_score(self, prices: pd.Series) -> pd.Series:
        """Get liquidity score based on price range"""
        
        scores = np.full(len(prices), 50)  # Default score
        
        # Price ranges with different liquidity characteristics
        scores = np.where((prices >= 8000) & (prices <= 15000), 80, scores)   # High liquidity
        scores = np.where((prices >= 15000) & (prices <= 25000), 70, scores)  # Good liquidity
        scores = np.where((prices >= 25000) & (prices <= 40000), 60, scores)  # Moderate liquidity
        scores = np.where(prices < 8000, 40, scores)                          # Lower liquidity
        scores = np.where(prices > 40000, 30, scores)                         # Lower liquidity
        
        return pd.Series(scores, index=prices.index)
    
    def calculate_market_trends_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market trends score (0-100)"""
        
        # Base trend score
        trend_scores = np.full(len(data), 50)
        
        # Make-specific trend adjustments
        for make in data['make'].unique():
            make_mask = data['make'] == make
            make_adjustment = self.get_make_trend_adjustment(make)
            trend_scores[make_mask] += make_adjustment
        
        # Year-specific adjustments (newer cars trending up)
        current_year = datetime.now().year
        age_adjustment = np.clip((data['year'] - (current_year - 10)) * 2, -20, 20)
        trend_scores += age_adjustment
        
        # Fuel type trends
        fuel_adjustment = data.apply(self.get_fuel_type_adjustment, axis=1)
        trend_scores += fuel_adjustment
        
        return np.clip(trend_scores, 0, 100)
    
    def get_make_trend_adjustment(self, make: str) -> float:
        """Get trend adjustment for specific makes"""
        
        trend_adjustments = {
            'Toyota': 15,      # Strong positive trend
            'Honda': 10,       # Positive trend
            'Lexus': 8,        # Luxury positive trend
            'Mazda': 5,        # Moderate positive
            'Subaru': 3,       # Slight positive
            'Nissan': 0,       # Neutral
            'Mitsubishi': -5,  # Slight negative
            'Infiniti': -8     # Luxury segment challenges
        }
        
        return trend_adjustments.get(make, 0)
    
    def get_fuel_type_adjustment(self, row) -> float:
        """Get adjustment based on fuel type trends"""
        
        fuel_type = row.get('fuel_type', '').lower()
        
        adjustments = {
            'hybrid': 20,      # Very positive trend
            'electric': 15,    # Strong positive trend
            'petrol': 0,       # Neutral
            'diesel': -10,     # Negative trend in UK
            'lpg': -5          # Declining trend
        }
        
        for fuel, adjustment in adjustments.items():
            if fuel in fuel_type:
                return adjustment
        
        return 0
    
    def calculate_weighted_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate final weighted score"""
        
        final_score = (
            data['profitability_score'] * self.weights['profitability'] +
            data['market_demand_score'] * self.weights['market_demand'] +
            data['risk_assessment_score'] * self.weights['risk_assessment'] +
            data['liquidity_score'] * self.weights['liquidity'] +
            data['market_trends_score'] * self.weights['market_trends']
        )
        
        return np.clip(final_score, 0, 100)
    
    def add_percentile_rankings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add percentile rankings for key metrics"""
        
        metrics_to_rank = [
            'final_score', 'profitability_score', 'market_demand_score',
            'risk_assessment_score', 'liquidity_score'
        ]
        
        for metric in metrics_to_rank:
            if metric in data.columns:
                data[f'{metric}_percentile'] = data[metric].rank(pct=True) * 100
        
        return data
    
    def get_score_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'D'
    
    def get_investment_category(self, score: float) -> str:
        """Categorize investment opportunity"""
        
        if score >= 85:
            return 'Premium Opportunity'
        elif score >= 75:
            return 'Strong Opportunity'
        elif score >= 65:
            return 'Good Opportunity'
        elif score >= 55:
            return 'Moderate Opportunity'
        elif score >= 45:
            return 'Marginal Opportunity'
        else:
            return 'Poor Opportunity'
    
    def generate_top_recommendations(self, scored_data: pd.DataFrame, top_n: int = 20) -> Dict:
        """Generate top vehicle recommendations"""
        
        if scored_data.empty:
            return {}
        
        top_vehicles = scored_data.head(top_n)
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'total_analyzed': len(scored_data),
            'top_vehicles': [],
            'category_breakdown': {},
            'summary_insights': {}
        }
        
        # Top vehicles details
        for _, vehicle in top_vehicles.iterrows():
            vehicle_info = {
                'rank': len(recommendations['top_vehicles']) + 1,
                'make': vehicle['make'],
                'model': vehicle['model'],
                'year': int(vehicle['year']),
                'final_score': round(vehicle['final_score'], 1),
                'score_grade': vehicle['score_grade'],
                'investment_category': vehicle['investment_category'],
                'profit_margin': vehicle['profit_margin'],
                'expected_profit': vehicle['gross_profit'],
                'roi': vehicle['roi'],
                'risk_level': self.get_risk_level(vehicle['risk_score']),
                'risk_score': vehicle['risk_score'],
                'estimated_sale_days': int(vehicle.get('avg_days_listed', 30)),
                'uk_market_price': vehicle['uk_avg_price'],
                'japan_total_cost': vehicle['japan_avg_total_cost'],
                'market_demand': round(vehicle['demand_score'], 1),
                'sample_confidence': 'High' if vehicle['uk_sample_size'] >= 10 else 'Moderate' if vehicle['uk_sample_size'] >= 5 else 'Low'
            }
            recommendations['top_vehicles'].append(vehicle_info)
        
        # Category breakdown
        category_counts = scored_data['investment_category'].value_counts()
        recommendations['category_breakdown'] = category_counts.to_dict()
        
        # Summary insights
        recommendations['summary_insights'] = {
            'highest_scoring_vehicle': {
                'vehicle': f"{top_vehicles.iloc[0]['make']} {top_vehicles.iloc[0]['model']} ({int(top_vehicles.iloc[0]['year'])})",
                'score': round(top_vehicles.iloc[0]['final_score'], 1)
            },
            'most_profitable': {
                'vehicle': f"{scored_data.loc[scored_data['profit_margin'].idxmax(), 'make']} {scored_data.loc[scored_data['profit_margin'].idxmax(), 'model']}",
                'margin': scored_data['profit_margin'].max()
            },
            'fastest_selling': {
                'vehicle': f"{scored_data.loc[scored_data['avg_days_listed'].idxmin(), 'make']} {scored_data.loc[scored_data['avg_days_listed'].idxmin(), 'model']}",
                'days': int(scored_data['avg_days_listed'].min())
            },
            'average_profit_margin': scored_data['profit_margin'].mean(),
            'average_roi': scored_data['roi'].mean(),
            'total_profit_potential': scored_data['gross_profit'].sum()
        }
        
        return recommendations
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        
        if risk_score <= 2:
            return 'Very Low'
        elif risk_score <= 4:
            return 'Low'
        elif risk_score <= 6:
            return 'Moderate'
        elif risk_score <= 8:
            return 'High'
        else:
            return 'Very High'
    
    def analyze_score_distribution(self, scored_data: pd.DataFrame) -> Dict:
        """Analyze the distribution of scores"""
        
        if scored_data.empty:
            return {}
        
        return {
            'score_statistics': {
                'mean': round(scored_data['final_score'].mean(), 1),
                'median': round(scored_data['final_score'].median(), 1),
                'std': round(scored_data['final_score'].std(), 1),
                'min': round(scored_data['final_score'].min(), 1),
                'max': round(scored_data['final_score'].max(), 1)
            },
            'grade_distribution': scored_data['score_grade'].value_counts().to_dict(),
            'category_distribution': scored_data['investment_category'].value_counts().to_dict(),
            'component_averages': {
                'profitability': round(scored_data['profitability_score'].mean(), 1),
                'market_demand': round(scored_data['market_demand_score'].mean(), 1),
                'risk_assessment': round(scored_data['risk_assessment_score'].mean(), 1),
                'liquidity': round(scored_data['liquidity_score'].mean(), 1),
                'market_trends': round(scored_data['market_trends_score'].mean(), 1)
            }
        }
    
    def generate_make_model_rankings(self, scored_data: pd.DataFrame) -> Dict:
        """Generate rankings by make and model"""
        
        if scored_data.empty:
            return {}
        
        # Make rankings
        make_scores = scored_data.groupby('make').agg({
            'final_score': 'mean',
            'profit_margin': 'mean',
            'roi': 'mean',
            'risk_score': 'mean'
        }).round(2)
        
        make_rankings = make_scores.sort_values('final_score', ascending=False)
        
        # Top models overall
        model_rankings = scored_data.groupby(['make', 'model']).agg({
            'final_score': 'mean',
            'profit_margin': 'mean',
            'roi': 'mean'
        }).round(2).sort_values('final_score', ascending=False).head(10)
        
        return {
            'make_rankings': make_rankings.to_dict('index'),
            'top_models': model_rankings.to_dict('index'),
            'analysis_date': datetime.now().isoformat()
        }
    
    # PORTFOLIO OPTIMIZATION - COMPLETE IMPLEMENTATION
    def calculate_portfolio_optimization(self, scored_data: pd.DataFrame, budget: float = 100000) -> Dict:
        """Calculate optimal portfolio allocation given a budget - COMPLETE METHOD"""
        
        self.logger.info(f"Starting portfolio optimization with budget £{budget:,.0f}")
        
        if scored_data.empty or budget <= 0:
            self.logger.warning("No data or invalid budget for portfolio optimization")
            return {
                'budget': budget,
                'allocated': 0,
                'remaining': budget,
                'portfolio': [],
                'expected_total_profit': 0,
                'portfolio_roi': 0,
                'number_of_vehicles': 0,
                'utilization_rate': 0,
                'message': 'No data available or invalid budget'
            }
        
        try:
            # Log available columns for debugging
            self.logger.info(f"Available columns: {list(scored_data.columns)}")
            
            # Make a copy to avoid modifying original data
            opportunities = scored_data.copy()
            
            # Sort by final score (best first)
            if 'final_score' in opportunities.columns:
                opportunities = opportunities.sort_values('final_score', ascending=False)
            else:
                self.logger.warning("No final_score column, using original order")
            
            # Ensure we have required columns with fallbacks
            self._ensure_portfolio_columns(opportunities, budget)
            
            # Filter to affordable vehicles
            affordable = opportunities[
                (opportunities['vehicle_cost'] > 0) & 
                (opportunities['vehicle_cost'] <= budget)
            ].copy()
            
            self.logger.info(f"Found {len(affordable)} affordable vehicles out of {len(opportunities)} total")
            
            if affordable.empty:
                return {
                    'budget': budget,
                    'allocated': 0,
                    'remaining': budget,
                    'portfolio': [],
                    'expected_total_profit': 0,
                    'portfolio_roi': 0,
                    'number_of_vehicles': 0,
                    'utilization_rate': 0,
                    'message': f'No vehicles found within budget of £{budget:,.0f}'
                }
            
            # Build portfolio using greedy approach
            portfolio = []
            remaining_budget = budget
            total_expected_profit = 0
            
            # Limit to top 15 vehicles to avoid overly complex portfolios
            for _, vehicle in affordable.head(15).iterrows():
                vehicle_cost = float(vehicle['vehicle_cost'])
                
                if vehicle_cost <= remaining_budget and len(portfolio) < 10:
                    try:
                        portfolio_item = {
                            'make': str(vehicle.get('make', 'Unknown')),
                            'model': str(vehicle.get('model', 'Vehicle')),
                            'year': int(vehicle.get('year', 2020)),
                            'investment': vehicle_cost,
                            'expected_profit': float(vehicle.get('expected_profit', 0)),
                            'roi': float(vehicle.get('roi_calc', 0)),
                            'score': float(vehicle.get('final_score', 50)),
                            'risk_score': float(vehicle.get('risk_score', 5))
                        }
                        
                        portfolio.append(portfolio_item)
                        remaining_budget -= vehicle_cost
                        total_expected_profit += portfolio_item['expected_profit']
                        
                        self.logger.info(f"Added to portfolio: {portfolio_item['make']} {portfolio_item['model']} - £{vehicle_cost:,.0f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error adding vehicle to portfolio: {e}")
                        continue
            
            # Calculate final metrics
            total_invested = budget - remaining_budget
            portfolio_roi = total_expected_profit / total_invested if total_invested > 0 else 0
            utilization_rate = total_invested / budget if budget > 0 else 0
            
            result = {
                'budget': budget,
                'allocated': total_invested,
                'remaining': remaining_budget,
                'portfolio': portfolio,
                'expected_total_profit': total_expected_profit,
                'portfolio_roi': portfolio_roi,
                'number_of_vehicles': len(portfolio),
                'utilization_rate': utilization_rate,
                'average_score': sum(v['score'] for v in portfolio) / len(portfolio) if portfolio else 0,
                'average_roi': sum(v['roi'] for v in portfolio) / len(portfolio) if portfolio else 0,
                'optimization_success': True
            }
            
            self.logger.info(f"Portfolio optimization completed: {len(portfolio)} vehicles, £{total_invested:,.0f} invested")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'budget': budget,
                'allocated': 0,
                'remaining': budget,
                'portfolio': [],
                'expected_total_profit': 0,
                'portfolio_roi': 0,
                'number_of_vehicles': 0,
                'utilization_rate': 0,
                'error': str(e),
                'optimization_success': False
            }
    
    def _ensure_portfolio_columns(self, df: pd.DataFrame, budget: float):
        """Ensure all required columns exist for portfolio optimization"""
        
        # Vehicle cost column
        if 'japan_avg_total_cost' in df.columns:
            df['vehicle_cost'] = pd.to_numeric(df['japan_avg_total_cost'], errors='coerce')
        elif 'uk_avg_price' in df.columns:
            df['vehicle_cost'] = pd.to_numeric(df['uk_avg_price'], errors='coerce') * 0.8  # Estimate
        else:
            df['vehicle_cost'] = budget * 0.15  # Default to 15% of budget
        
        # Fill NaN values in vehicle_cost
        df['vehicle_cost'] = df['vehicle_cost'].fillna(budget * 0.15)
        
        # Expected profit column
        if 'gross_profit' in df.columns:
            df['expected_profit'] = pd.to_numeric(df['gross_profit'], errors='coerce')
        else:
            df['expected_profit'] = df['vehicle_cost'] * 0.2  # 20% profit estimate
        
        # Fill NaN values in expected_profit
        df['expected_profit'] = df['expected_profit'].fillna(df['vehicle_cost'] * 0.2)
        
        # ROI calculation
        df['roi_calc'] = df['expected_profit'] / df['vehicle_cost']
        df['roi_calc'] = df['roi_calc'].fillna(0.2)  # Default 20% ROI
        
        # Ensure final_score exists
        if 'final_score' not in df.columns:
            df['final_score'] = 50.0  # Default score
        else:
            df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce').fillna(50.0)
        
        # Ensure other required columns
        if 'make' not in df.columns:
            df['make'] = 'Unknown'
        if 'model' not in df.columns:
            df['model'] = 'Vehicle'
        if 'year' not in df.columns:
            df['year'] = 2020
        if 'risk_score' not in df.columns:
            df['risk_score'] = 5.0
        
        self.logger.info("Portfolio columns ensured with fallback values")