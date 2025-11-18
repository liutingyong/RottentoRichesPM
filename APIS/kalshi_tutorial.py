import os
import sys
import time
import uuid
from dotenv import load_dotenv
from kalshi_python import Configuration, KalshiClient
from pathlib import Path
from cryptography.hazmat.primitives import serialization

sys.path.append(str(Path(__file__).parent.parent / "basic_processing_and_sentiment_analysis"))
from sentiment_classifier import sentiment_analyzer


def extract_ticker_from_url(url_or_ticker):
    if not any(domain in url_or_ticker.lower() for domain in ['kalshi.co', 'kalshi.com', 'http', 'www']):
        return url_or_ticker
    
    if 'kalshi.co' in url_or_ticker or 'kalshi.com' in url_or_ticker:
        if '/trade/' in url_or_ticker:
            ticker = url_or_ticker.split('/trade/')[-1]
        elif '/markets/' in url_or_ticker:
            path_part = url_or_ticker.split('/markets/')[-1]
            ticker = path_part.split('/')[-1].upper() if '/' in path_part else path_part.upper()
        elif '/event/' in url_or_ticker:
            ticker = url_or_ticker.split('/event/')[-1].upper()
        else:
            ticker = url_or_ticker.rstrip('/').split('/')[-1].upper()
        
        ticker = ticker.split('?')[0].split('#')[0]
        return ticker
    
    return url_or_ticker


# Global config for order placement
client_config = None

def init_client():
    global client_config
    load_dotenv()
    host = os.getenv('KALSHI_URL')
    api_key_id = os.getenv('KALSHI_API_KEY_ID')
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')
    
    config = Configuration(host=host)
    with open(private_key_path, "r") as f:
        private_key = f.read()
    
    config.api_key_id = api_key_id
    config.private_key_pem = private_key
    client_config = config
    return KalshiClient(config)


def get_single_market(client, ticker):
    try:
        market = client.get_market(ticker)
        if hasattr(market, 'market'):
            market = market.market
        return {
            'ticker': market.ticker if hasattr(market, 'ticker') else None,
            'yes_ask': market.yes_ask if hasattr(market, 'yes_ask') and market.yes_ask else 0,
            'no_ask': market.no_ask if hasattr(market, 'no_ask') and market.no_ask else 0,
            'status': market.status if hasattr(market, 'status') else None,
        }
    except:
        return None


def get_event_markets(client, event_ticker):
    try:
        markets = client.get_markets(event_ticker=event_ticker, limit=100, status='open')
        if hasattr(markets, 'markets'):
            return [m.ticker for m in markets.markets if hasattr(m, 'ticker')]
        return []
    except Exception as e:
        print(f"API error: {str(e)}")
        return []


def analyze_and_bet(client, market_tickers, sentiment_data):
    sentiment_results = sentiment_analyzer.analyze_multiple_texts(sentiment_data)
    sentiment_percentage = sentiment_results['positive_percentage']
    print(f"Sentiment: {sentiment_percentage:.0%} positive")
    
    for ticker in market_tickers:
        print(f"\nEvaluating {ticker}...")
        market_data = get_single_market(client, ticker)
        if not market_data:
            print(f"  ⏸️  Skipping {ticker}: could not fetch market data")
            continue
        
        # Log the status for debugging
        status = market_data.get('status')
        if status:
            print(f"  Status: {status}")
        
        yes_ask = market_data.get('yes_ask') or 0
        no_ask = market_data.get('no_ask') or 0
        print(f"  Prices: YES={yes_ask}¢ NO={no_ask}¢")
        
        if sentiment_percentage > 0.6:
            if 0 < yes_ask < 50:
                print(f"  ✅ Recommend betting YES (sentiment {sentiment_percentage:.0%} > 60%)")
                confirm = input(f"  Place order? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    if place_market_order(client, ticker, 'yes', 1):
                        print(f"  ✅ YES order placed on {ticker}")
                else:
                    print(f"  ⏸️  Order cancelled")
            else:
                print(f"  ⏸️  Not betting YES (price {yes_ask}¢ too high)")
        elif sentiment_percentage < 0.4:
            if 0 < no_ask < 50:
                print(f"  ✅ Recommend betting NO (sentiment {sentiment_percentage:.0%} < 40%)")
                confirm = input(f"  Place order? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    if place_market_order(client, ticker, 'no', 1):
                        print(f"  ✅ NO order placed on {ticker}")
                else:
                    print(f"  ⏸️  Order cancelled")
            else:
                print(f"  ⏸️  Not betting NO (price {no_ask}¢ too high)")
        else:
            print(f"  ⏸️  No bet (neutral sentiment {sentiment_percentage:.0%})")


def place_market_order(client, ticker, side, amount):
    try:
        # Get current prices
        market = client.get_market(ticker)
        if hasattr(market, 'market'):
            market = market.market
        
        current_price = market.yes_ask if side == 'yes' and hasattr(market, 'yes_ask') else market.no_ask if side == 'no' and hasattr(market, 'no_ask') else 50
        order_price = int(current_price) + 1  # Add 1 cent to improve fill
        
        # Create order using PortfolioApi
        from kalshi_python import PortfolioApi, ApiClient
        
        order_data = {
            "ticker": ticker,
            "client_order_id": f"order_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            "side": side,
            "action": "buy",
            "count": amount,
            "type": "limit"
        }
        
        if side == "yes":
            order_data["yes_price"] = order_price
        else:
            order_data["no_price"] = order_price
        
        # Submit the order using PortfolioApi
        from kalshi_python import PortfolioApi
        
        # Use PortfolioApi - access via the client's configuration 
        portfolio_api = PortfolioApi(client.api_client)
        response = portfolio_api.create_order(**order_data)
        
        if response and hasattr(response, 'order'):
            print(f"  ✅ Order placed successfully")
            print(f"  Order ID: {response.order.order_id if hasattr(response.order, 'order_id') else 'N/A'}")
            return True
        else:
            print(f"  ❌ Order failed: Invalid response")
            return False
            
    except Exception as e:
        print(f"  ❌ Order failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    load_dotenv()
    event_ticker = os.getenv('EVENT_TICKER')
    if not event_ticker:
        print("ERROR: EVENT_TICKER not set in .env file")
        return
    
    print(f"Analyzing event: {event_ticker}")
    
    client = init_client()
    print("Connected to Kalshi API")
    
    market_tickers = get_event_markets(client, event_ticker)
    if not market_tickers:
        print(f"No markets found for {event_ticker}")
        return
    
    print(f"Found {len(market_tickers)} markets")
    
    scraped_data_dir = Path("../../webscraping/scraped_data")
    sentiment_data = [file.read_text(encoding="utf-8") for file in scraped_data_dir.glob("*.txt")] if scraped_data_dir.exists() else ["Sample text"]
    
    analyze_and_bet(client, market_tickers, sentiment_data)


if __name__ == "__main__":
    main()
