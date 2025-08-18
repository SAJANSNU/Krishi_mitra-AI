"""
Dynamic Market Data API Module
Integrates with multiple Indian agricultural market APIs for real-time price data
Supports APMC, eNAM, AgMarkNet, and other government sources
"""

import os
import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass
from config.settings import DATA_GOV_API_KEY, MARKET_API_ENDPOINT
from src.utils.logger import log

@dataclass
class MarketPrice:
    commodity: str
    market: str
    state: str
    district: str
    price: float
    date: str
    unit: str = "quintal"
    grade: Optional[str] = None

@dataclass
class PriceTrend:
    commodity: str
    current_price: float
    price_change: float
    price_change_percent: float
    trend_direction: str
    price_history: List[float]

class MarketAPI:
    def __init__(self):
        self.api_key = os.getenv("DATA_GOV_API_KEY", DATA_GOV_API_KEY)
        self.base_endpoints = {
            "agmarknet": "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
            "enam": "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24",
            "commodity_prices": "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Krishi-Mitra-AI/1.0',
            'Accept': 'application/json'
        })
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        log.info("Market API initialized with dynamic endpoints")

    def get_commodity_prices(self, commodity: Optional[str] = None, 
                           state: Optional[str] = None,
                           district: Optional[str] = None,
                           market: Optional[str] = None,
                           limit: int = 100,
                           days_back: int = 7) -> Dict[str, Any]:
        """
        Fetch commodity prices with dynamic filtering capabilities.
        
        Args:
            commodity: Crop/commodity name (e.g., "Tomato", "Wheat")
            state: State name for filtering
            district: District name for filtering  
            market: Specific market name
            limit: Maximum number of records
            days_back: Number of days to look back for data
            
        Returns:
            Dictionary with price data and metadata
        """
        try:
            # Check cache first
            cache_key = f"{commodity}_{state}_{district}_{market}_{limit}_{days_back}"
            if self._is_cache_valid(cache_key):
                log.info(f"Returning cached data for {cache_key}")
                return self.cache[cache_key]

            # Build dynamic parameters
            params = self._build_market_params(commodity, state, district, market, limit, days_back)
            
            # Try multiple endpoints dynamically
            for endpoint_name, endpoint_url in self.base_endpoints.items():
                try:
                    log.info(f"Trying endpoint: {endpoint_name}")
                    response = self._make_api_request(endpoint_url, params)
                    
                    if response and self._validate_response(response):
                        processed_data = self._process_market_response(response, commodity)
                        
                        # Cache successful response
                        self.cache[cache_key] = processed_data
                        log.info(f"Successfully fetched data from {endpoint_name}")
                        return processed_data
                        
                except Exception as e:
                    log.warning(f"Endpoint {endpoint_name} failed: {e}")
                    continue
            
            # If all endpoints fail, return empty structure
            return self._empty_market_response("No data available from any endpoint")
            
        except Exception as e:
            log.error(f"Market API request failed: {e}")
            return self._empty_market_response(str(e))

    def get_price_trends(self, commodity: str, days: int = 30) -> Dict[str, Any]:
        """
        Calculate price trends for a commodity over specified days.
        
        Args:
            commodity: Commodity name
            days: Number of days for trend analysis
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            # Get historical data
            historical_data = self.get_commodity_prices(
                commodity=commodity, 
                limit=200, 
                days_back=days
            )
            
            if not historical_data.get("records"):
                return {"error": f"No historical data found for {commodity}"}
            
            # Process trend analysis
            trend_analysis = self._calculate_trends(historical_data["records"], commodity)
            return trend_analysis
            
        except Exception as e:
            log.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}

    def compare_markets(self, commodity: str, markets: List[str] = None) -> Dict[str, Any]:
        """
        Compare prices across different markets for a commodity.
        
        Args:
            commodity: Commodity name
            markets: List of market names to compare
            
        Returns:
            Dictionary with market comparison data
        """
        try:
            if not markets:
                # Get data from all available markets
                all_data = self.get_commodity_prices(commodity=commodity, limit=500)
                markets = list(set([record.get("market", "Unknown") 
                                  for record in all_data.get("records", [])]))
            
            comparison_data = {}
            for market in markets:
                market_data = self.get_commodity_prices(
                    commodity=commodity, 
                    market=market, 
                    limit=50
                )
                
                if market_data.get("records"):
                    avg_price = self._calculate_average_price(market_data["records"])
                    comparison_data[market] = {
                        "average_price": avg_price,
                        "record_count": len(market_data["records"]),
                        "latest_date": self._get_latest_date(market_data["records"])
                    }
            
            return {
                "commodity": commodity,
                "market_comparison": comparison_data,
                "best_price_market": self._find_best_price_market(comparison_data),
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            log.error(f"Market comparison failed: {e}")
            return {"error": str(e)}

    def get_market_insights(self, commodity: str) -> Dict[str, Any]:
        """
        Generate comprehensive market insights for a commodity.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Dictionary with market insights and recommendations
        """
        try:
            # Get current prices
            current_data = self.get_commodity_prices(commodity=commodity, limit=100)
            
            # Get price trends
            trend_data = self.get_price_trends(commodity=commodity, days=30)
            
            # Get market comparison
            comparison_data = self.compare_markets(commodity=commodity)
            
            # Generate insights
            insights = self._generate_market_insights(
                current_data, trend_data, comparison_data, commodity
            )
            
            return insights
            
        except Exception as e:
            log.error(f"Market insights generation failed: {e}")
            return {"error": str(e)}

    def _build_market_params(self, commodity: Optional[str], state: Optional[str], 
                           district: Optional[str], market: Optional[str], 
                           limit: int, days_back: int) -> Dict[str, Any]:
        """Build dynamic API parameters based on input filters."""
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": str(limit)
        }
        
        # Add filters dynamically
        if commodity:
            # Handle different commodity name formats
            commodity_variants = self._get_commodity_variants(commodity)
            params["filters[commodity]"] = commodity_variants[0]  # Use primary variant
        
        if state:
            params["filters[state]"] = state
            
        if district:
            params["filters[district]"] = district
            
        if market:
            params["filters[market]"] = market
        
        # Add date filtering if supported
        if days_back > 0:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            params["filters[arrival_date]"] = f">={start_date.strftime('%Y-%m-%d')}"
        
        return params

    def _get_commodity_variants(self, commodity: str) -> List[str]:
        """Get different name variants for a commodity."""
        # Common variations in Indian market data
        commodity_mapping = {
            "tomato": ["Tomato", "TOMATO", "tomato"],
            "onion": ["Onion", "ONION", "onion"],
            "potato": ["Potato", "POTATO", "potato"],
            "wheat": ["Wheat", "WHEAT", "wheat"],
            "rice": ["Rice", "RICE", "rice", "Paddy", "PADDY"],
            "cotton": ["Cotton", "COTTON", "cotton"],
            "sugarcane": ["Sugarcane", "SUGARCANE", "sugarcane"],
            "maize": ["Maize", "MAIZE", "maize", "Corn"],
            "soybean": ["Soybean", "SOYBEAN", "soybean", "Soya bean"],
            "chili": ["Chili", "CHILI", "chili", "Chilli", "Red Chilli"]
        }
        
        commodity_lower = commodity.lower()
        return commodity_mapping.get(commodity_lower, [commodity, commodity.upper(), commodity.lower()])

    def _make_api_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request with retry logic and error handling."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                return data
                
            except requests.exceptions.Timeout:
                log.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    
            except requests.exceptions.RequestException as e:
                log.warning(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    
            except ValueError as e:
                log.warning(f"JSON decode failed: {e}")
                break
        
        return None

    def _validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate API response structure."""
        if not isinstance(response, dict):
            return False
            
        # Check for common error indicators
        if "error" in response or "Error" in response:
            return False
            
        # Check for records
        if "records" not in response:
            return False
            
        return len(response.get("records", [])) > 0

    def _process_market_response(self, response: Dict[str, Any], 
                                commodity: Optional[str]) -> Dict[str, Any]:
        """Process and standardize market API response."""
        records = response.get("records", [])
        processed_records = []
        
        for record in records:
            try:
                processed_record = self._standardize_market_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                log.warning(f"Failed to process record: {e}")
                continue
        
        # Calculate statistics
        statistics = self._calculate_market_statistics(processed_records, commodity)
        
        return {
            "records": processed_records,
            "total_results": len(processed_records),
            "statistics": statistics,
            "commodity_filter": commodity,
            "data_source": "government_apis",
            "last_updated": datetime.now().isoformat()
        }

    def _standardize_market_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize individual market record format."""
        try:
            # Handle different field name variations
            price_fields = ["modal_price", "Modal_x0020_Price", "price", "rate"]
            commodity_fields = ["commodity", "Commodity", "item_name"]
            market_fields = ["market", "Market", "market_name"]
            state_fields = ["state", "State"]
            district_fields = ["district", "District"]
            date_fields = ["arrival_date", "date", "Date", "price_date"]
            
            # Extract price
            price = None
            for field in price_fields:
                if field in record and record[field]:
                    try:
                        price = float(str(record[field]).replace(",", ""))
                        break
                    except (ValueError, TypeError):
                        continue
            
            if price is None or price <= 0:
                return None
            
            # Extract other fields
            commodity = self._extract_field(record, commodity_fields)
            market = self._extract_field(record, market_fields)
            state = self._extract_field(record, state_fields)
            district = self._extract_field(record, district_fields)
            date = self._extract_field(record, date_fields)
            
            return {
                "commodity": commodity or "Unknown",
                "market": market or "Unknown",
                "state": state or "Unknown", 
                "district": district or "Unknown",
                "price": price,
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "unit": "quintal",  # Standard unit
                "raw_record": record  # Keep original for debugging
            }
            
        except Exception as e:
            log.warning(f"Record standardization failed: {e}")
            return None

    def _extract_field(self, record: Dict[str, Any], field_names: List[str]) -> Optional[str]:
        """Extract field value from record using multiple possible field names."""
        for field_name in field_names:
            if field_name in record and record[field_name]:
                value = str(record[field_name]).strip()
                if value and value.lower() != "null":
                    return value
        return None

    def _calculate_market_statistics(self, records: List[Dict[str, Any]], 
                                   commodity: Optional[str]) -> Dict[str, Any]:
        """Calculate market statistics from processed records."""
        if not records:
            return {}
        
        prices = [record["price"] for record in records if record["price"]]
        
        if not prices:
            return {}
        
        statistics = {
            "average_price": sum(prices) / len(prices),
            "min_price": min(prices),
            "max_price": max(prices),
            "price_range": max(prices) - min(prices),
            "total_markets": len(set(record["market"] for record in records)),
            "total_states": len(set(record["state"] for record in records)),
            "record_count": len(records),
            "commodity": commodity
        }
        
        # Calculate median
        sorted_prices = sorted(prices)
        n = len(sorted_prices)
        if n % 2 == 0:
            statistics["median_price"] = (sorted_prices[n//2-1] + sorted_prices[n//2]) / 2
        else:
            statistics["median_price"] = sorted_prices[n//2]
        
        return statistics

    def _calculate_trends(self, records: List[Dict[str, Any]], 
                         commodity: str) -> Dict[str, Any]:
        """Calculate price trends from historical data."""
        try:
            # Sort records by date
            dated_records = []
            for record in records:
                try:
                    date_str = record.get("date", "")
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    dated_records.append((date_obj, record["price"]))
                except (ValueError, TypeError):
                    continue
            
            if len(dated_records) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            dated_records.sort(key=lambda x: x[0])
            
            # Calculate trend
            prices = [price for _, price in dated_records]
            dates = [date for date, _ in dated_records]
            
            # Simple trend calculation
            start_price = prices[0]
            end_price = prices[-1]
            price_change = end_price - start_price
            price_change_percent = (price_change / start_price) * 100 if start_price > 0 else 0
            
            trend_direction = "stable"
            if abs(price_change_percent) > 5:
                trend_direction = "increasing" if price_change > 0 else "decreasing"
            
            return {
                "commodity": commodity,
                "start_date": dates[0].isoformat(),
                "end_date": dates[-1].isoformat(),
                "start_price": start_price,
                "end_price": end_price,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "trend_direction": trend_direction,
                "data_points": len(prices),
                "price_history": prices[-10:]  # Last 10 prices
            }
            
        except Exception as e:
            log.error(f"Trend calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_average_price(self, records: List[Dict[str, Any]]) -> float:
        """Calculate average price from records."""
        prices = [record["price"] for record in records if record.get("price", 0) > 0]
        return sum(prices) / len(prices) if prices else 0.0

    def _get_latest_date(self, records: List[Dict[str, Any]]) -> str:
        """Get the latest date from records."""
        dates = []
        for record in records:
            try:
                date_str = record.get("date", "")
                if date_str:
                    dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            except ValueError:
                continue
        
        return max(dates).isoformat() if dates else datetime.now().isoformat()

    def _find_best_price_market(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find market with best (highest) price for selling."""
        if not comparison_data:
            return {}
        
        best_market = None
        best_price = 0
        
        for market, data in comparison_data.items():
            price = data.get("average_price", 0)
            if price > best_price:
                best_price = price
                best_market = market
        
        return {
            "market": best_market,
            "price": best_price
        } if best_market else {}

    def _generate_market_insights(self, current_data: Dict[str, Any], 
                                trend_data: Dict[str, Any],
                                comparison_data: Dict[str, Any], 
                                commodity: str) -> Dict[str, Any]:
        """Generate comprehensive market insights."""
        insights = {
            "commodity": commodity,
            "current_market_status": {},
            "price_trends": {},
            "market_recommendations": [],
            "analysis_date": datetime.now().isoformat()
        }
        
        # Current market status
        if current_data.get("statistics"):
            stats = current_data["statistics"]
            insights["current_market_status"] = {
                "average_price": stats.get("average_price", 0),
                "price_range": f"₹{stats.get('min_price', 0):.2f} - ₹{stats.get('max_price', 0):.2f}",
                "markets_covered": stats.get("total_markets", 0),
                "data_freshness": "latest_available"
            }
        
        # Price trends
        if not trend_data.get("error"):
            insights["price_trends"] = {
                "direction": trend_data.get("trend_direction", "stable"),
                "change_percent": trend_data.get("price_change_percent", 0),
                "change_amount": trend_data.get("price_change", 0)
            }
        
        # Recommendations
        recommendations = []
        
        if trend_data.get("trend_direction") == "increasing":
            recommendations.append("Prices are trending upward. Consider holding if storage is available.")
        elif trend_data.get("trend_direction") == "decreasing":
            recommendations.append("Prices are declining. Consider selling soon.")
        else:
            recommendations.append("Prices are stable. Normal selling timing applies.")
        
        if comparison_data.get("best_price_market"):
            best_market = comparison_data["best_price_market"]
            recommendations.append(f"Best selling price found in {best_market.get('market', 'N/A')}")
        
        insights["market_recommendations"] = recommendations
        
        return insights

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get("cache_timestamp", 0)
        return (time.time() - cache_time) < self.cache_duration

    def _empty_market_response(self, error_message: str) -> Dict[str, Any]:
        """Return empty response structure with error message."""
        return {
            "records": [],
            "total_results": 0,
            "statistics": {},
            "error": error_message,
            "data_source": "api_error",
            "last_updated": datetime.now().isoformat()
        }

    def get_api_status(self) -> Dict[str, Any]:
        """Get API health status."""
        status = {
            "api_key_configured": bool(self.api_key),
            "endpoints": {},
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat()
        }
        
        # Test each endpoint
        for name, url in self.base_endpoints.items():
            try:
                test_params = {"api-key": self.api_key, "format": "json", "limit": "1"}
                response = self._make_api_request(url, test_params)
                status["endpoints"][name] = {
                    "status": "operational" if response else "failed",
                    "url": url
                }
            except Exception as e:
                status["endpoints"][name] = {
                    "status": "error",
                    "error": str(e),
                    "url": url
                }
        
        return status
