import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_autocorrelation(series, max_lag):
    autocorrelations = [series.autocorr(lag=i) for i in range(max_lag+1)]

    # Plotting the autocorrelations
    plt.figure(figsize=(10, 5))
    plt.bar(range(max_lag+1), autocorrelations, color='dodgerblue')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of h_prev up to {} lags'.format(max_lag))
    plt.show()

    return autocorrelations

def get_time_to_expire(h_prev):
    h_prev_series = pd.Series(h_prev)
    autocorrelations = plot_autocorrelation(h_prev_series, max_lag=90)

    # Find the first lag where autocorrelation changes from positive to negative
    for i in range(1, len(autocorrelations)):
        if autocorrelations[i-1] > 0 and autocorrelations[i] < 0:
            print(f"Lag where autocorrelation first changes from positive to negative: {i}")
            return i

    # If no change from positive to negative is found, find the minimum autocorrelation
    lag_to_min_autocorr = np.argmin(np.abs(autocorrelations))
    print(f"No positive to negative transition; minimum autocorrelation at lag: {lag_to_min_autocorr}")
    return lag_to_min_autocorr

# # Example usage:
# h_prev = np.random.normal(0, 1, 100)  # Generate some random data to test
# time_to_expire = get_time_to_expire(h_prev)

