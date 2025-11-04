from kalshi_python import Configuration, KalshiClient
from cryptography.hazmat.primitives import serialization


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