import json
import os
from pprint import pprint
import requests

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = "0b07354f7148495aa33ec87b5a7ba195"
endpoint = "https://api.bing.microsoft.com/bing/v7.0/search"

# Query term(s) to search for.
query = "pirate image portrait"

# Construct a request
mkt = 'en-US'
params = {'q': query, 'mkt': mkt}
headers = {'Ocp-Apim-Subscription-Key': subscription_key}

# Call the API
try:
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()

    print("Headers:")
    print(response.headers)

    print("JSON Response:")
    pprint(response.json())
except Exception as ex:
    raise ex
