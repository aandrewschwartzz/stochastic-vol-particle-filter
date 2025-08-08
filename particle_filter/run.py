
from run_functions.baseline.get_benchmark_data import get_benchmark
from run_functions.baseline.get_sim_data import get_sim_data

from algo.ABC_pg_CAPF import ABC_pg_cAPF
import pandas as pd
import numpy as np
import time
import concurrent.futures

import random
import sys
import csv
import scipy.stats
import matplotlib.pyplot as plt

from run_functions.time_plus_one.expectation_prediction import forecast_next_ht_and_sign


def run_sim(end_date_str, alpha, beta, pg_params, seed=None): 

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df, observations = get_sim_data(
        end_date_str,
        )
    
    burn_in, sample, particles, epsilon = pg_params

    Z_params = [1.725, 0.0915, 1, 0]

    Z_params = [alpha, beta, 1, 0]

    abc_pg_capf = ABC_pg_cAPF(observations, burn_in, sample, particles, epsilon, Z_params)
    # cProfile.runctx('abc_pg_capf.run()', globals(), locals(), filename=None)
    start_time = time.time() 
    #Never tell me the odds 
    h_prev, tau, phi, sigma = abc_pg_capf.run()
    end_time = time.time()
    duration = end_time - start_time
    print(f"The test took {duration} seconds to complete.")
    sys.stdout.flush()

    # h_prev = h_prev.to_list()

    h_prev = np.concatenate([np.array([np.nan]), h_prev])
    h_t = h_prev[-1]
    print(h_t)
    h_t_plus_one, dir = forecast_next_ht_and_sign(h_t, tau, phi)


    results = {
            'tau':tau,
            'phi': phi,
            'sigma':sigma,
            'h_t_plus_one': h_t_plus_one,
            'dir': dir

        }

    return results

import concurrent.futures

def collect_results(end_date_str_list, alpha, beta, pg_params):
    results = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map end dates to future objects
        futures = {executor.submit(run_sim, end_date, alpha, beta, pg_params): end_date for end_date in end_date_str_list}
        
        # Process completed futures
        for future in concurrent.futures.as_completed(futures):
            end_date_str = futures[future]
            try:
                result = future.result()
                if result:
                    # Add results to dictionary under their respective end date
                    results.setdefault(end_date_str, []).append(result)
                else:
                    print(f"No result returned for {end_date_str}")
            except Exception as exc:
                print(f'Generated an exception in collecting results for {end_date_str}: {exc}')

    return results

def dict_to_csv(data, csv_filename):

    flattened_data = []
    for date, values_list in data.items():
        for values in values_list:
            row = {'date': date}
            row.update(values)
            flattened_data.append(row)

    # Get all column headers
    fieldnames = ['date'] + list(flattened_data[0].keys())[1:] if flattened_data else []

    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened_data:
            writer.writerow(row)

# This adjusted version of dict_to_csv now directly handles a list of dictionaries under each date.


def experiment(start_date_str, end_date_str, alpha, beta, pg_params):

    data, dates = get_benchmark(start_date_str, end_date_str)

    prediction_values = collect_results(dates[1:], alpha, beta, pg_params)
    # Convert prediction_values to a DataFrame for easier merging
    predictions_df = pd.DataFrame([
        {'date': date, 'dir': preds[0]['dir']}
        for date, preds in prediction_values.items()
    ])
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date

    # Merge the prediction DataFrame into the original data DataFrame
    data = data.merge(predictions_df, left_index=True, right_on='date', how='left').set_index('date')
    data.rename(columns={'dir': 'Directional Volatility Prediction'}, inplace=True)
    print(data)

    #dict_to_csv(prediction_values, f'Backtest_{end_date_str}.csv')

    return prediction_values


if __name__ == "__main__":

    start_date_str, end_date_str = '2024-06-01', '2024-06-05'
    alpha, beta = 1.725, 0.0915

    burn_in = 100
    sample = 200
    particles = 500
    epsilon = 0.00001

    pg_params = [burn_in, sample, particles, epsilon]

    results = experiment(
                        start_date_str, 
                        end_date_str, 
                        alpha, beta, 
                        pg_params)
    print(results)
