
from run_functions.baseline.get_benchmark_data import get_benchmark
from run_functions.baseline.get_sim_data import get_sim_data

from run_functions.time_plus_one.autocorrelation import get_time_to_expire


from algo.ABC_pg_CAPF import ABC_pg_cAPF
import pandas as pd
import numpy as np
import time
import concurrent.futures

import random
import sys
import csv


from run_functions.time_plus_one.cond_probs import find_window_and_signal

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
    h_prev, tau, phi, sigma, conditional_probabilities = abc_pg_capf.run()

    best_window_size, cond_vol_up, tail_value = find_window_and_signal(conditional_probabilities)
    h_t_plus_one, pred_vol_up = forecast_next_ht_and_sign(h_prev[-1], tau)
    slices_to_exp = get_time_to_expire(h_prev)

    vol_up = None

    if (cond_vol_up and pred_vol_up) or (not cond_vol_up and not pred_vol_up):
        vol_up = cond_vol_up  # Since both are the same, we can use either


    end_time = time.time()
    duration = end_time - start_time
    print(f"The test took {duration} seconds to complete.")
    sys.stdout.flush()

    # h_prev = h_prev.to_list()

    h_prev = np.concatenate([np.array([np.nan]), h_prev])

    print(tau, phi, sigma)

    results = {}


    results['no_weighting'] = {
            'tau':tau,
            'phi': phi,
            'sigma':sigma,
            'vol_up': vol_up,
            'tail_est': tail_value, 
            'exp_slices': slices_to_exp,

            # 'expiry_date': date,
        }

    return results

import concurrent.futures

def collect_results(end_date_str_list, alpha, beta, pg_params):
    results = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit a future for each date without specifying a seed
        futures = {executor.submit(run_sim, end_date, alpha, beta, pg_params): end_date for end_date in end_date_str_list}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            end_date_str = futures[future]
            try:
                result = future.result()
                if result:
                    # exp_weighting = result['exp_weighting']
                    no_weighting = result['no_weighting']

                    if end_date_str not in results:
                        results[end_date_str] = {
                            # 'exp_weighting': [], 
                                                 'no_weighting': []
                                                 }

                    # results[end_date_str]['exp_weighting'].append(exp_weighting)
                    results[end_date_str]['no_weighting'].append(no_weighting)
                else:
                    print(f"No result returned for {end_date_str}")

            except Exception as exc:
                print(f'Generated an exception for {end_date_str}: {exc}')

    return results

def dict_to_csv(data, csv_filename):
    # Flatten the dictionary
    flattened_data = []
    for date, inner_dict in data.items():
        for strategy, values_list in inner_dict.items():
            for values in values_list:
                row = {'date': date}
                row.update(values)
                flattened_data.append(row)

    # Get all column headers
    fieldnames = ['date'] + list(flattened_data[0].keys())[1:]

    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened_data:
            writer.writerow(row)

def experiment(start_date_str, end_date_str, alpha, beta, pg_params):

    data, dates = get_benchmark(start_date_str, end_date_str,)
    
    prediction_values = collect_results(dates[1:], alpha, beta, pg_params)
    dict_to_csv(prediction_values, f'Backtest_{end_date_str}.csv')

    return prediction_values


if __name__ == "__main__":
    start_date_str, end_date_str = '2021-05-01', '2022-05-01'
    alpha, beta = 1.725, 0.0915

    burn_in = 500
    sample = 800
    particles = 800
    epsilon = 0.00001

    pg_params = [burn_in, sample, particles, epsilon]

    results = experiment(start_date_str, end_date_str, alpha, beta, pg_params)
    print(results)
