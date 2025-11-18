import sys, time, uuid, glob
from pathlib import Path
from kalshi_python import Configuration, KalshiClient, PortfolioApi
from cryptography.hazmat.primitives import serialization
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


# Configure the client
config = Configuration(
    host="https://api.elections.kalshi.com/trade-api/v2"
)

# For authenticated requests
# Read private key from file
with open("KalshiProdAPI.pem", "r") as f:
    private_key = f.read()

config.api_key_id = "72192223-a094-49fd-aff6-bf1d720b901a"
config.private_key_pem = private_key

# Initialize the client
client = KalshiClient(config)

# Make API calls
balance = client.get_balance()
print(f"Balance: ${balance.balance / 100:.2f}")

series_ticker = "KXRTAVATARFIREANDASH"
events_response = client.get_events(series_ticker=series_ticker)

# XGBoost analyzer on new reviews
project_root = Path(__file__).parent.parent
train_dir = project_root / "webscaping" / "review_data" / "training" / "scraped_data"
new_dir = project_root / "webscaping" / "review_data" / "new_reviews" / "scraped_data"
urls_file = project_root / "webscaping" / "review_data" / "training" / "urls.txt"

# Load training data and labels
train_paths = glob.glob(str(train_dir / "*.txt"))
new_paths = glob.glob(str(new_dir / "*.txt"))

# Extract labels from urls.txt
def parse_labels(urls_file_path):
    url_to_label, current_label = {}, None
    with open(urls_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if line in ['0', '1']: current_label = int(line)
            elif line.startswith(('http://', 'https://')) and current_label is not None:
                url_to_label[line] = current_label
    return url_to_label

def extract_url(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('URL: '): return first_line[5:]
    except: pass
    return None

url_to_label = parse_labels(urls_file)
y = []
matched_paths = []
for path in train_paths:
    url = extract_url(path)
    if url and url in url_to_label:
        y.append(url_to_label[url])
        matched_paths.append(path)
train_paths = matched_paths

if len(y) == 0 or len(train_paths) != len(y):
    print(f"Warning: {len(train_paths)} files but {len(y)} labels. Using default labels.")
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0] + 31 * [0] + 27 * [1]
    if len(train_paths) != len(y):
        y = y[:len(train_paths)] if len(y) > len(train_paths) else y + [0] * (len(train_paths) - len(y))

pipe = Pipeline([("tfidf", TfidfVectorizer(input='filename', stop_words='english', lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, max_features=30000)), ("xgb", XGBClassifier(objective='binary:logistic', n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method='hist', eval_metric='logloss', n_jobs=-1))])
pipe.fit(train_paths, y)
sentiment = pipe.predict_proba(new_paths)[:, 1].mean() if new_paths else 0.5
print(f"Sentiment: {sentiment:.0%} positive\n")

# Market Strategy: Expected Value Optimization
# Strategy: Find the market with best risk-adjusted expected value
# EV = (Probability of Win × Payout) - Cost
# We bet when: EV > threshold AND price is reasonable (< 70¢)
opportunities = []
for event in events_response.events:
    markets = client.get_markets(event_ticker=event.event_ticker, status='open', limit=100).markets
    for m in markets:
        yes_ask, no_ask = (m.yes_ask or 0), (m.no_ask or 0)
        if yes_ask > 0 and no_ask > 0:
            # Calculate expected value for YES (if sentiment is positive)
            yes_ev = (sentiment * (100 - yes_ask)) - yes_ask if yes_ask < 70 else -999
            # Calculate expected value for NO (if sentiment is negative)  
            no_ev = ((1 - sentiment) * (100 - no_ask)) - no_ask if no_ask < 70 else -999
            # Score: EV * confidence (how far from 50/50)
            yes_score = yes_ev * abs(sentiment - 0.5) if sentiment > 0.55 and yes_ev > 5 else -999
            no_score = no_ev * abs(sentiment - 0.5) if sentiment < 0.45 and no_ev > 5 else -999
            if yes_score > 0: opportunities.append((m.ticker, "yes", yes_ask, yes_score, yes_ev))
            if no_score > 0: opportunities.append((m.ticker, "no", no_ask, no_score, no_ev))

# Pick best opportunity (highest EV)
if opportunities:
    best = max(opportunities, key=lambda x: x[4])  # Sort by EV instead of score
    ticker, side, price, score, ev = best
    print(f"Best opportunity: {ticker} ({side.upper()})")
    print(f"  Price: {price}¢ | Expected Value: {ev:.1f}¢ per contract")
    if input(f"Place {side.upper()} order? (Expected profit: {ev:.1f}¢) (yes/no): ").strip().lower() == 'yes':
        market = client.get_market(ticker).market
        order_data = {"ticker": ticker, "client_order_id": f"order_{int(time.time())}_{uuid.uuid4().hex[:8]}", "side": side, "action": "buy", "count": 1, "type": "limit", f"{side}_price": int(price) + 1}
        PortfolioApi(client.api_client).create_order(**order_data)
        print(f"✅ Order placed on {ticker}")
else:
    print("No good opportunities found (need strong sentiment + good prices)")


