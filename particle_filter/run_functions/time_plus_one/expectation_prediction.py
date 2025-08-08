import numpy as np

def forecast_next_ht_and_sign(h_t, tau, phi):
    # Calculate log(h_t) using the model equation
    log_ht = tau + phi * np.log(h_t)
    
    # Calculate h_t for the next time step by exponentiating log(h_t)
    h_t_plus_one = np.exp(log_ht)
    
    # Determine the sign of change in volatility
    dir = np.sign(h_t_plus_one - h_t )
    return h_t_plus_one, dir