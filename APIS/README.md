# Kalshi API Tutorial

A minimal tutorial for the Kalshi Python SDK, focusing on **Markets** operations.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your `.env` File

Create a `.env` file with your credentials:

```env
KALSHI_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_API_KEY_ID=your-api-key-id-here
KALSHI_PRIVATE_KEY_PATH=path/to/private_key.pem
```

Get your credentials from: https://kalshi.com/api

### 3. Run the Example

```bash
python example.py
```

## What It Does

The example demonstrates three key operations:
1. **Get Balance** - Check your account balance
2. **Get Markets** - List open markets
3. **Get Market Details** - Get detailed info for a specific market (commented out)

## Common Market Operations

### Get All Open Markets
```python
markets = client.get_markets(status='open', limit=100)
```

### Get a Specific Market
```python
market = client.get_market('TICKER-HERE')
print(f"Title: {market.market.title}")
print(f"Yes Bid: {market.market.yes_bid}¢")
```

### Get Market Orderbook
```python
orderbook = client.get_market_orderbook('TICKER-HERE', depth=10)
```

### Filter Markets
```python
# By event
markets = client.get_markets(event_ticker='PRES.2024', status='open')

# By series
markets = client.get_markets(series_ticker='SERIES-TICKER', status='open')
```

### Get Recent Trades
```python
trades = client.get_trades(ticker='TICKER-HERE', limit=20)
```

## Documentation

- Full API docs: https://docs.kalshi.com/python-sdk/api
- Python SDK: https://pypi.org/project/kalshi-python/
