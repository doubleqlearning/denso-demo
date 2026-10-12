# src/tools/available_tools.py
from .fetch_from_coinmarket import fetch_from_coinmarketcap

# Ánh xạ tên hàm (chuỗi) với đối tượng hàm thực tế
function_mapping = {
    "fetch_from_coinmarketcap": fetch_from_coinmarketcap,
}

# Danh sách các schema của tool để gửi cho LLM
tools_schema = [{
    "type": "function",
    "function": {
        "name": "fetch_from_coinmarketcap",
        "description": "Get the latest price in USD for a list of cryptocurrency symbols.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of cryptocurrency symbols (e.g., ['BTC', 'ETH']).",
                }
            },
            "required": ["symbols"],
        },
    }
}]