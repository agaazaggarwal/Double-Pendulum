import numpy as np
from scipy.signal import savgol_filter

def smooth_trajectory(data, window=5):
    if len(data) < 3:
        return data
    if window % 2 == 0:
        window += 1
    window = min(window, len(data) - 1 if len(data) % 2 == 0 else len(data))
    if window < 3:
        return data
    try:
        return savgol_filter(data, window, 2, axis=0)
    except:
        return data

def unwrap_angle(angles):
    return np.unwrap(angles)