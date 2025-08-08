import pandas as pd

def check_window_size(conditional_probabilities, window_size, target_percentage):
    rolling_probs = pd.Series(conditional_probabilities).rolling(window=window_size, min_periods=1).mean()
    
    above_threshold = (rolling_probs > 0.5).sum() / len(rolling_probs)
    below_threshold = (rolling_probs < 0.5).sum() / len(rolling_probs)


    
    if above_threshold >= target_percentage:
        
        return True, False, rolling_probs.iloc[-window_size:].mean()
    elif below_threshold >= target_percentage:
        return True, True, rolling_probs.iloc[-window_size:].mean()
    else:
        return False, None, None

# Iterate through window sizes
def find_window_and_signal(conditional_probabilities):
    best_window_size = None
    for window_size in range(1, 5000):
        meets_criteria, vol_up, average = check_window_size(conditional_probabilities, window_size, 0.95)
        if meets_criteria:
            best_window_size = window_size
            break  # Stop as soon as we find a window size that meets either criteria
    if best_window_size is not None: 
        return best_window_size, vol_up, average
