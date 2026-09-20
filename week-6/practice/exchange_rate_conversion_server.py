from dotenv import load_dotenv
from fastmcp import FastMCP
import requests
import os

load_dotenv()
EXCHANGE_RATE_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

mcp = FastMCP("Exchange Rate Conversion")

@mcp.tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Get the currency conversion rate between a base currency and target currency.
    Use this tool first when a user asks to convert currencies. Pass its
    conversion rate result to the convert tool.
    """
    
    url = f'https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_KEY}/pair/{base_currency}/{target_currency}'
    
    raw_result = requests.get(url)  
    data = raw_result.json()
    print(data)
    return data['conversion_rate']

@mcp.tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """
    Convert a currency amount using a conversion rate obtained from
    get_conversion_factor. Use this tool after getting the conversion rate.
    """
    return base_currency_value * conversion_rate

# if __name__ == '__main__':
#     mcp.run(transport='http', port=8000)

if __name__ == '__main__':
    mcp.run()