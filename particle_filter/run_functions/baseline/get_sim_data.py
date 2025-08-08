import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt

def get_sim_data(end_date_str=None, slices_to_subtract=400):
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

    if end_date_str is None:
        end_date_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Prepare date range
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date = end_date - timedelta(days=slices_to_subtract)
    start_date_str = start_date.strftime('%Y-%m-%d')

    # Fetch and prepare data for '^GSPC'
    ticker_data = yf.Ticker('^GSPC', session=session)
    data = None
    try:
        data = ticker_data.history(period='1d', start=start_date_str, end=end_date_str, interval='1d')
        data = data[['Close', 'High', 'Low']].rename(columns={'Close': 'GSPC Close'})
        data.index = data.index.tz_convert('UTC').date  # Convert index to UTC and strip time
        data['GSPC Close Log Diff'] = np.log(data['GSPC Close']).diff()  # Calculate log differences
    except requests.exceptions.RequestException as e:
        print(f"Request error for ticker ^GSPC: {e}")

    # Calculate log differences for observations if data is fetched
    observations = np.diff(np.log(data['GSPC Close']).to_numpy())[1:] if data is not None else np.array([])

    return data, observations
# # Example usage
# end_date_str = '2023-01-08'  # Example end date

# df, observations = get_sim_data(end_date_str=end_date_str)
# print(df.tail())

# plt.figure(figsize=(12, 6))  # Set the figure size for better readability

# # Plot 'GSPC Close'
# plt.plot(df.index, df['GSPC Close'], label='GSPC Close', color='green')
# plt.xlabel('Date')  # Label for the X-axis
# plt.ylabel('GSPC Close', color='green')  # Label for the Y-axis with matching color
# plt.tick_params(axis='y', labelcolor='green')  # Color the y-axis label to match the line color
# plt.legend(loc='upper right')  # Position the legend
# plt.title('GSPC Close Over Time')  # Title of the graph
# plt.grid(True)  # Turn on the grid
# plt.show()  # Display the plot

