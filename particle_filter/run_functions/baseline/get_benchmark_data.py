import yfinance as yf
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import numpy as np

def get_benchmark(start_date_str, end_date_str):
    # Convert start_date_str and end_date_str to datetime objects and then to date objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    # Calculate new start date that is 5 days before the original start date
    adjusted_start_date = start_date - timedelta(days=40)
    adjusted_start_date_str = adjusted_start_date.strftime('%Y-%m-%d')

    # Set up a requests session with retry logic
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Fetch data for the GSPC index (ticker symbol '^GSPC')
    gspc_data = yf.Ticker('^GSPC', session=session)
    try:
        data = gspc_data.history(start=adjusted_start_date_str, end=end_date_str)
    except requests.exceptions.SSLError as e:
        print(f"SSL error: {e}")
        return None, []
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None, []

    # Filter and rename columns
    data = data[['Open', 'Close']].rename(columns={'Close': 'GSPC Close', 'Open': 'GSPC Open'})

    # Convert index to UTC and strip time
    data.index = data.index.tz_convert('UTC').date
    data['Daily Returns'] = data['GSPC Close'].pct_change()
    data['21-Day Volatility'] = data['Daily Returns'].rolling(window=21).std()
    data['Volatility Change Sign'] = np.sign(data['21-Day Volatility'].diff())
    

    # Create a list of date strings formatted as 'YYYY-%m-%d' within the original start and end date range

    test_dates = [date for date in data.index if start_date <= date <= end_date]
    data = data[data.index.isin(test_dates)]

    test_dates = [date.strftime('%Y-%m-%d') for date in data.index]
    


    return data, test_dates

# # Test the function
# start_date_str, end_date_str = '2023-01-01', '2023-01-08'
# data, test_dates = get_benchmark(start_date_str, end_date_str)
# print(data)
# print(test_dates)