import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from IPython.display import display
from scipy.spatial.transform import Rotation as R
import os, joblib
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score,  recall_score
import torch
import polars as pl
from pathlib import Path
import inspect
import psutil
from scipy.signal import welch
from scipy.stats import entropy

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION --
# =============================================================================

class Config: 
    """Central configuration class for training and data parameters"""

    path = Path(os.getcwd())
    parts = list(path.parts)

    # Optionally remove the root slash if desired
    if parts[0] == '/':
        parts = parts[1:]

    if parts[0] == 'Users':
        # Paths for Kaggle environment
        TRAIN_PATH = "~/.kaggle/sensor-data/train.csv"
        TRAIN_DEMOGRAPHICS_PATH = "~/.kaggle/sensor-data/train_demographics.csv"
        TEST_PATH = "~/.kaggle/sensor-data/test.csv"
        TEST_DEMOGRAPHICS_PATH = "~/.kaggle/sensor-data/test_demographics.csv"
        EXPORT_DIR =  "/Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/data"                                  
        EXPORT_MODELS_PATH =  "/Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/models" 
        OPTUNA_PATH_SAVED =   "/Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/optuna_data" 
        OPTUNA_PATH_LOGS =   "/Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/codes/logs_optuna" 
        EXPORT_PARAMS_DIR =   "/Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/codes/trials" 
 
    
    elif parts[0] == 'home':
        TRAIN_PATH = "/home/mathieuisoard/remote-github/sensor-data/train.csv"
        TRAIN_DEMOGRAPHICS_PATH = "/home/mathieuisoard/remote-github/sensor-data/train_demographics.csv"
        TEST_PATH = "/home/mathieuisoard/remote-github/sensor-data/test.csv"
        TEST_DEMOGRAPHICS_PATH = "/home/mathieuisoard/remote-github/sensor-data/test_demographics.csv"
        EXPORT_DIR =  "/home/mathieuisoard/remote-github/data"                                  
        EXPORT_MODELS_PATH =  "/home/mathieuisoard/remote-github/models_from_gcloud" 
        OPTUNA_PATH_SAVED =   "/home/mathieuisoard/remote-github/optuna_data" 
        OPTUNA_PATH_LOGS =   "/home/mathieuisoard/remote-github/codes/logs_optuna" 
        EXPORT_PARAMS_DIR =   "/home/mathieuisoard/remote-github/codes/trials" 

    else:
        print("NEW ROOT DIRECTORY") 


    os.makedirs(EXPORT_DIR, exist_ok=True)                                 
    os.makedirs(EXPORT_MODELS_PATH, exist_ok=True)                                 
    os.makedirs(OPTUNA_PATH_SAVED, exist_ok=True)                                 
    os.makedirs(OPTUNA_PATH_LOGS, exist_ok=True) 
    os.makedirs(EXPORT_PARAMS_DIR, exist_ok=True) 

    # Training parameters
    SEED = 42
    N_FOLDS = 5
    PERCENTILE = 95
    PADDING = 127

    # Feature columns
    ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
    ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    
# Set reproducibility
np.random.seed(Config.SEED)

def check_gpu_availability():

    import torch
    if torch.backends.mps.is_available():
        #print("MPS (Apple GPU) is available.")
        return 'mps'
    else:
        #print("MPS not available. Using CPU.")
        return 'cpu'

# Check GPU availability
DEVICE = torch.device(check_gpu_availability())

print(f"✓ Configuration loaded for Kaggle environment (Device: {DEVICE})")


def clean_data(data_sequences, cols, prefix = 'both'):
    
    if prefix == 'both':
        print("removing tof and thm missing data columns from sequences! Saving seq_id in a dic with cols to remove")
        tof_and_thm_cols = [col for col in cols if (col.startswith('thm') or col.startswith('tof')) ]
    else: 
        print(f"removing {prefix} missing data columns from sequences! Saving seq_id in a dic with cols to remove")
        tof_and_thm_cols = [col for col in cols if col.startswith(prefix) ]

    tof_thm_nan_prefixes = {}
    for sequence_id, sequence_data in data_sequences:
        nan_cols = sequence_data[tof_and_thm_cols].columns[sequence_data[tof_and_thm_cols].isna().any()]
        if nan_cols.any():
            if (prefix == 'both' or prefix == 'tof'):
                prefixes = set(col.rsplit("_", 1)[0] for col in nan_cols if col.startswith('tof'))
            else:
                prefixes = set()
            if (prefix == 'both' or prefix == 'thm'):
                prefixes.update(set(col for col in nan_cols if col.startswith('thm')))

            tof_thm_nan_prefixes[sequence_id] = prefixes
            cols_to_drop = [col for col in sequence_data.columns if any(col.startswith(p) for p in prefixes)]
            sequence_data = sequence_data.drop(columns=cols_to_drop)
    print(f"found {len(tof_thm_nan_prefixes)} sequences with missing data")
    return data_sequences, tof_thm_nan_prefixes

def handle_missing_values_quaternions(quaternion):
    quat_clean = quaternion.copy()
    
    number_of_nan = quaternion.isna().sum(axis = 1)
    rows_with_0_nan = number_of_nan == 0
    rows_with_1_nan = number_of_nan == 1
    rows_with_N_nan = number_of_nan > 1

    ### normalize quaternions to 1 when no NaN has been detected 
    quat_values = quaternion.loc[rows_with_0_nan].values
    norms = np.linalg.norm(quat_values, axis = 1)
    normalized_quats = np.zeros_like(quat_values)
    ## for non-zero norm, normalize to 1  
    nonzero_norms = norms > 1e-6
    normalized_quats[nonzero_norms] = quat_values[nonzero_norms] / norms[nonzero_norms, np.newaxis]
    ## for zero-norm, normalize to the unit quaternion
    normalized_quats[~nonzero_norms] = [1.0, 0.0, 0.0, 0.0]
    ##update quaternion DataFrame
    quat_clean.loc[rows_with_0_nan] = normalized_quats

    ###handle 1 missing value 
    #use |w|² + |x|² + |y|² + |z|² = 1
    if len(quaternion[rows_with_1_nan].index.tolist()) > 0:
        nan_columns_per_row = quaternion[rows_with_1_nan].isna().idxmax(axis=1)
        unnorm_quat = quaternion[rows_with_1_nan].pow(2).sum(axis =1, skipna = True)
        vals = np.sqrt(np.maximum(0, 1 - unnorm_quat))
        for row, col, val in zip(unnorm_quat.index, nan_columns_per_row, vals):
            if row > 0:
                if quat_clean.loc[row - 1, col] >= 0:
                    quat_clean.loc[row, col] = val
                else:
                    quat_clean.loc[row, col] = -val
            else:
                next_row = row + 1
                # Go forward until a non-NaN is found or reach the end
                while next_row < len(quat_clean) and np.isnan(quat_clean.loc[next_row, col]):
                    next_row += 1
                if next_row == len(quat_clean):
                    quat_clean.loc[rows_with_1_nan] = [0, 0, 0, 0]
                    quat_clean.loc[rows_with_1_nan, 'rot_w'] = 1
                    break
                else:
                    if quat_clean.loc[next_row, col] >= 0:
                        quat_clean.loc[row, col] = val
                    else:
                        quat_clean.loc[row, col] = -val
    quat_clean.loc[rows_with_N_nan] = [0, 0, 0, 0]
    quat_clean.loc[rows_with_N_nan, 'rot_w'] = 1
    return quat_clean

def check_missing_values_quaternion(data_sequences):
    seq_id_quaternion_nan = []
    check_norm_quaternion = []
    for seq_id, data_sequence in data_sequences:
        quaternion_cols = [col for col in data_sequence.columns if col.startswith('rot_')]
        nan_quat_cols = data_sequence[quaternion_cols].columns[data_sequence[quaternion_cols].isna().any()]
        normalize_quat = data_sequence[quaternion_cols].pow(2).sum(axis = 1).mean()
        if nan_quat_cols.any():
            #print(data_sequence[[col for col in data_sequence.columns if col.startswith('acc_')]])
            seq_id_quaternion_nan.append(seq_id)
        if (not nan_quat_cols.any()) and normalize_quat < 0.99:
            check_norm_quaternion.append(seq_id)
    print(f"✓ number of seq_id with missing values in quaternion: {len(seq_id_quaternion_nan)}")
    print(f"✓ number of unnormalized quaternions for complete quaternions: {len(check_norm_quaternion)}")
    return seq_id_quaternion_nan


def regularize_quaternions_per_sequence(data_sequence):
    data_clean = data_sequence.copy()
    quaternion_cols = [col for col in data_sequence.columns if col.startswith('rot_')]
    nan_quat_cols = data_sequence[quaternion_cols].columns[data_sequence[quaternion_cols].isna().any()]
    normalize_quat = data_sequence[quaternion_cols].pow(2).sum(axis = 1).mean()  
    if nan_quat_cols.any():
        data_clean[quaternion_cols] = handle_missing_values_quaternions(data_sequence[quaternion_cols])
    if (not nan_quat_cols.any()) and normalize_quat < 0.99:
        data_clean[quaternion_cols] = handle_missing_values_quaternions(data_sequence[quaternion_cols])

    ### Check failed regularization
    nan_quat_cols_clean = data_clean[quaternion_cols].columns[data_clean[quaternion_cols].isna().any()]
    normalize_quat_clean = data_clean[quaternion_cols].pow(2).sum(axis = 1).mean() 
    if nan_quat_cols_clean.any():
        print("!!NaN values have been detected after regularisation!!")
    if (not nan_quat_cols_clean.any()) and normalize_quat_clean < 0.99:
        print("!!Not normalized quaternions have been detected after regularisation!!")
    return data_clean



def clean_and_check_quaternion(data):
    data_clean = data.copy()
    data_sequences = data_clean.groupby('sequence_id')
    seq_id_quaternion_nan = check_missing_values_quaternion(data_sequences)
    if len(seq_id_quaternion_nan) > 0:
        for seq_id in seq_id_quaternion_nan:
            data_sequence = data_sequences.get_group(seq_id)
            idx = data_sequence.index  # Get the index of the group
            quaternion_cols = [col for col in data_sequence.columns if col.startswith('rot_')]
            # Apply quaternion cleaning function
            data_clean.loc[idx, quaternion_cols] = handle_missing_values_quaternions(data_sequence[quaternion_cols])
    ##Check quaternion
        data_sequences = data_clean.groupby('sequence_id')
        print("")
        print(" --- missing values in quaternions have been handled ---")
        check_missing_values_quaternion(data_sequences)
        print("")
    return data_clean

def compute_acceleration_features(sequence_data, demographics):
    sequence_data_with_acc = sequence_data.copy()
    correct_rot_order = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    correct_acc_order = ['acc_x', 'acc_y', 'acc_z']
    col_acc_world = ['acc_x_world', 'acc_y_world', 'acc_z_world']
    col_linear_acc = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
    col_X_world = ['X_world_x', 'X_world_y', 'X_world_z']
    col_Y_world = ['Y_world_x', 'Y_world_y', 'Y_world_z']
    col_Z_world = ['Z_world_x', 'Z_world_y', 'Z_world_z']
    remove_gravity = [0, 0, 9.81]
    
    data_rot = sequence_data[correct_rot_order]
    data_acc = sequence_data[correct_acc_order]
    sensor_x = np.zeros( data_acc.to_numpy().shape )
    sensor_y = np.zeros( data_acc.to_numpy().shape )
    sensor_z = np.zeros( data_acc.to_numpy().shape )
    sensor_x[:, 0] = 1
    sensor_y[:, 1] = 1
    sensor_z[:, 2] = 1
    data_rot_scipy = data_rot.to_numpy() 

    try:
        r = R.from_quat(data_rot_scipy)
        sequence_data_with_acc[col_acc_world] = pd.DataFrame(r.apply(data_acc.to_numpy()) - remove_gravity)
        sequence_data_with_acc[col_X_world] = pd.DataFrame(r.apply(sensor_x))
        sequence_data_with_acc[col_Y_world] = pd.DataFrame(r.apply(sensor_y))
        sequence_data_with_acc[col_Z_world] = pd.DataFrame(r.apply(sensor_z))
        
        gravity_in_sensor = r.apply(remove_gravity, inverse=True)
        acc_raw = sequence_data_with_acc[correct_acc_order].values
        linear_acc = acc_raw - gravity_in_sensor
        sequence_data_with_acc[col_linear_acc] = linear_acc

    except ValueError:
        print("Warning: world accelerations failed using device accelerations, replace by device acc data")
        sequence_data_with_acc[col_linear_acc] = sequence_data_with_acc[correct_acc_order]
        sequence_data_with_acc[col_acc_world] = sequence_data_with_acc[correct_acc_order]
        sequence_data_with_acc[col_X_world] = sequence_data_with_acc[correct_acc_order]
        sequence_data_with_acc[col_Y_world] = sequence_data_with_acc[correct_acc_order]
        sequence_data_with_acc[col_Z_world] = sequence_data_with_acc[correct_acc_order]

    sequence_data_with_acc['acc_norm_world'] =sequence_data_with_acc[col_acc_world].apply(np.linalg.norm, axis=1)
    sequence_data_with_acc['acc_norm'] =sequence_data_with_acc[correct_acc_order].apply(np.linalg.norm, axis=1)
    sequence_data_with_acc['linear_acc_norm'] =sequence_data_with_acc[col_linear_acc].apply(np.linalg.norm, axis=1)
    sequence_data_with_acc['acc_norm_jerk'] = sequence_data_with_acc['acc_norm'].diff().fillna(0)
    sequence_data_with_acc['linear_acc_norm_jerk'] =  sequence_data_with_acc['linear_acc_norm'].diff().fillna(0)

    subject = sequence_data['subject'].iloc[0]
    handedness = demographics[demographics['subject'] == subject]['handedness'].iloc[0] ## (0): left, (1): right
    if handedness == 0:
        sequence_data_with_acc['acc_x'] = - sequence_data_with_acc['acc_x'] #+ (-0.8526133780336856 + 0.3518238644621146)
        sequence_data_with_acc['linear_acc_x'] = - sequence_data_with_acc['linear_acc_x'] #+ (-0.8526133780336856 + 0.3518238644621146)

    return sequence_data_with_acc

def compute_angular_features(sequence_data, demographics, time_delta = 10):
    sequence_data_with_ang_vel = sequence_data.copy()
    correct_rot_order = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    quats = sequence_data[correct_rot_order].values

    rotations = R.from_quat(quats)
    rotvecs = rotations.as_rotvec()
    sequence_data_with_ang_vel[['rotvec_x', 'rotvec_y', 'rotvec_z']] = rotvecs
    sequence_data_with_ang_vel['angle_rad'] =  sequence_data_with_ang_vel[['rotvec_x', 'rotvec_y', 'rotvec_z']].apply(np.linalg.norm, axis=1)
    rot_diff = sequence_data_with_ang_vel[['rotvec_x', 'rotvec_y', 'rotvec_z']].diff().fillna(0)
    sequence_data_with_ang_vel['angular_speed'] = rot_diff.pow(2).sum(axis=1).pow(0.5)
    sequence_data_with_ang_vel['rot_angle'] = 2 * np.arccos(sequence_data['rot_w'].clip(-1, 1))
    sequence_data_with_ang_vel['rot_angle_vel'] = sequence_data_with_ang_vel['rot_angle'].diff().fillna(0)
    
    n_samples = quats.shape[0]
    ang_vel = np.zeros( (n_samples, 3))
    ang_dist = np.zeros(n_samples)

    for i in range(n_samples - 1):
        q1 = quats[i]
        q2 = quats[i + 1]

        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)):
            continue

        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Relative rotation from q1 to q2
            delta_r = r1.inv() * r2

            # Angle of rotation (in radians)
            ang_vel[i, : ] =  delta_r.as_rotvec()/time_delta
            ang_dist[i] = np.linalg.norm(delta_r.as_rotvec())
        except ValueError:
            pass

    sequence_data_with_ang_vel[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = ang_vel
    sequence_data_with_ang_vel['ang_dist'] = ang_dist

    subject = sequence_data['subject'].iloc[0]
    handedness = demographics[demographics['subject'] == subject]['handedness'].iloc[0] ## (0): left, (1): right
    if handedness == 0:
        sequence_data_with_ang_vel['rotvec_x'] = - sequence_data_with_ang_vel['rotvec_x'] #+ (-0.8526133780336856 + 0.3518238644621146)
        sequence_data_with_ang_vel['ang_vel_y'] = - sequence_data_with_ang_vel['ang_vel_y'] #+ (-0.8526133780336856 + 0.3518238644621146)
        sequence_data_with_ang_vel['ang_vel_z'] = - sequence_data_with_ang_vel['ang_vel_z'] #+ (-0.8526133780336856 + 0.3518238644621146)

    return sequence_data_with_ang_vel

def fft_gesture(signal):
    """
    Compute the normalized power in a band around a target frequency.

    Parameters:
    - signal: 1D array-like signal
    - freq: frequency of interest (Hz)
    - sampling_rate: sampling rate in Hz
    - bandwidth_ratio: fraction of freq to define integration window (e.g., 0.05 for ±5%)

    Returns:
    - normalized_band_power: power in [freq ± bandwidth] / total power
    """
    signal = np.asarray(signal)
    n = len(signal)
    #freqs = np.fft.rfftfreq(n, d=1./sampling_rate)
    fft_vals = np.fft.rfft(signal)
    power_spectrum = np.abs(fft_vals)**2 / n
    return power_spectrum / np.sum(power_spectrum)

def compute_fft_features(sequence_data):
    sequence_data_fft = sequence_data.copy()
    fft_to_compute = [
        'acc_x', 'acc_y', 'acc_z',
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
        'rotvec_x', 'rotvec_y', 'rotvec_z',
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
        'acc_norm', 'angle_rad'
    ]
    check_quat = ['rot_x', 'rot_y', 'rot_z']
    phase_gesture = sequence_data['phase_adj'] == 1    
    for feat in fft_to_compute:
        signal = sequence_data.loc[phase_gesture, feat].to_numpy()
        if sequence_data[check_quat].apply(np.linalg.norm, axis=1).mean() < 1e-6:
            signal_fft_pad = np.zeros_like(sequence_data[feat])
        else:
            signal_fft = fft_gesture( (signal - np.mean(signal))/np.std(signal) )
            signal_fft_pad =np.pad(signal_fft, (0, len(phase_gesture) - len(signal_fft)), 'constant')
        sequence_data_fft[f'{feat}_FFT'] = signal_fft_pad
    
    return sequence_data_fft

def get_angles(time_series, world_coord = False):
    theta, phi = [], []
    acc_features = ['acc_norm', 'acc_x', 'acc_y', 'acc_z']
    f_phi, f_theta = 'phi', 'theta'
    if world_coord:
        add_name = '_world'
        acc_features = [f + add_name for f in acc_features]
        f_phi, f_theta = f_phi + add_name, f_theta + add_name

    #numpy_time_series = time_series[acc_features].to_numpy()
    for a, ax, ay, az in zip(*time_series[acc_features].to_numpy().T):
        # Avoid division by zero
        # if a < 0:
        #     print(a)
        th = np.arccos(np.clip(az / (a + 1e-8), -1.0, 1.0))  # polar angle
        ph = np.arctan2(ay, ax)  # azimuthal angle
        theta.append(th)
        phi.append(ph)
    time_series[f_theta] = np.array(theta)
    time_series[f_phi] = np.array(phi)
    return time_series

def autocorr_frequency(signal, sampling_rate=1.0, min_lag=2, max_lag=None):
    """
    Estimate the dominant frequency in a signal using autocorrelation.

    Parameters:
    - signal: list or np.array of values
    - sampling_rate: Hz
    - min_lag: minimum lag to consider (to skip lag 0 and noise)
    - max_lag: optional max lag to consider

    Returns:
    - dominant_freq: float or None (in Hz)
    """
    signal = np.array(signal)
    if len(signal) < min_lag + 2:
        return 0.

    # Normalize and detrend
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only non-negative lags
    autocorr /= autocorr[0]  # Normalize

    # Define lag range to search
    if max_lag is None:
        max_lag = len(signal) #// 2

    search_range = autocorr[min_lag:max_lag]

    # Find peaks in the autocorrelation
    peaks, _ = find_peaks(search_range)

    if len(peaks) < 2:
        return 0.

    first_peak_lag = (peaks[-1] - peaks[0])/(len(peaks)-1)  # adjust for sliced lag
    period = first_peak_lag / sampling_rate
    freq = 1.0 / period

    return freq

def remove_frequency_component(signal, freq, sampling_rate, bandwidth=1.0, order=4):
    """
    Remove a specific frequency component using a Butterworth band-stop filter.

    Parameters:
    - signal: np.array of the signal values
    - freq: the target frequency to remove (Hz)
    - sampling_rate: the sampling rate of the signal (Hz)
    - bandwidth: the width of the stop band (Hz)
    - order: filter order (higher = steeper filter)

    Returns:
    - filtered_signal: the signal with the frequency component removed
    """
    bandwidth = 1 * freq
    nyquist = 0.5 * sampling_rate
    low = (freq - bandwidth / 2) / nyquist
    high = (freq + bandwidth / 2) / nyquist

    if low <= 0 or high >= 1:
        # Invalid range – don't apply filtering
        return signal.copy()

    # Create band-stop filter
    b, a = butter(order, [low, high], btype='bandstop')

    # Calculate required padding length
    padlen = 3 * max(len(a), len(b))

    if len(signal) <= padlen:
        # Too short for reliable filtering
        b, a = butter(order, [low, high], btype='bandstop')
        padlen = 3 * max(len(a), len(b))
        filtered_signal = filtfilt(b, a, signal, padlen=min(padlen, len(signal) - 1))
        # if len(signal) <= padlen:
        #     print("short")
        #     return signal.copy()
    else:
        filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def extract_freq_features(signal, fs=10):  # signal: [T, 3] for x,y,z IMU
    features = []
    for axis in range(signal.shape[1]):
        f, Pxx = welch(signal[:, axis], fs=fs, nperseg=fs)
        Pxx /= Pxx.sum()  # Normalize power spectrum

        centroid = np.sum(f * Pxx)
        entropy_val = entropy(Pxx)
        rolloff = f[np.where(np.cumsum(Pxx) >= 0.85)[0][0]]
        peak_freq = f[np.argmax(Pxx)]
        flatness = np.exp(np.mean(np.log(Pxx + 1e-8))) / (np.mean(Pxx) + 1e-8)

        features += [centroid, entropy_val, rolloff, peak_freq, flatness]

    return features  # [5 features x 3 axes = 15 features]

def sliding_window_freq_features(data_sequence, fs=10, window_size=10, stride=10):
    """
    data: (N, T, C) - batch of sequences
    returns: (N, T_new, F_freq)
    """
    names = ['centroid', 'entropy_val', 'rolloff', 'peak_freq', 'flatness']
    data_sequence_with_FFT = data_sequence.copy()
    signal = data_sequence[['acc_x', 'acc_y', 'acc_z']].to_numpy()
    T, _ = signal.shape
    features = []
    for i in range(0, T - window_size + 1, stride):
        window = signal[i:i+window_size, :]  # (N, w, C)
        f_list = extract_freq_features(window, fs=fs)  # for each sequence
        for j in range(window_size):
            features.append(f_list)  # (T, F_freq)
    print(np.array(features).shape)
    data_sequence_with_FFT[names] = np.array(features)
    # Stack over time: (T_new, N, F) → transpose → (N, T_new, F)
    return data_sequence_with_FFT


def compute_theta_phi_features(sequence_data):
    sequence_data_theta_phi = sequence_data.copy()
        
    sequence_data_theta_phi = get_angles(sequence_data_theta_phi)
    sequence_data_theta_phi = get_angles(sequence_data_theta_phi, world_coord=True)

    signal_phi = sequence_data_theta_phi['phi_world'].to_numpy()
    
    dym_zero_cross = np.zeros(len(signal_phi))
    window_size = 10
    for i in range(window_size, len(signal_phi)):
        window_phi = signal_phi[i-window_size: i]
        dym_zero_cross[i] = np.sum(np.diff(np.signbit(window_phi)).astype(int))

    sequence_data_theta_phi['zero_crossings_phi_dyn'] = dym_zero_cross
    return sequence_data_theta_phi

def compute_corr_and_svd_features(sequence_data):
    sequence_data_with_corr_and_svd = sequence_data.copy()

    svd_axis = [
        ['acc_x', 'acc_y', 'acc_z'],
        ['linear_acc_x', 'linear_acc_y', 'linear_acc_z'],
        ['rotvec_x', 'rotvec_y', 'rotvec_z'],
        ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']
    ]
    corr_features = [
        ('acc_x', 'acc_y'),
        ('acc_x', 'acc_z'),
        ('acc_y', 'acc_z'),
        ('linear_acc_x', 'linear_acc_y'),
        ('linear_acc_x', 'linear_acc_z'),
        ('linear_acc_y', 'linear_acc_z'),
        ('ang_vel_x', 'ang_vel_y'),
        ('ang_vel_x', 'ang_vel_z'),
        ('ang_vel_y', 'ang_vel_z'),
        ('rotvec_x', 'rotvec_y'),
        ('rotvec_x', 'rotvec_z'),
        ('rotvec_y', 'rotvec_z'),
        ('acc_norm', 'angle_rad'),
        ('acc_x', 'rotvec_x'),
        ('acc_x', 'rotvec_y'),
        ('acc_x', 'rotvec_z'),
        ('acc_y', 'rotvec_x'),
        ('acc_y', 'rotvec_y'),
        ('acc_y', 'rotvec_z'),
        ('acc_z', 'rotvec_x'),
        ('acc_z', 'rotvec_y'),
        ('acc_z', 'rotvec_z'),
        ('theta', 'phi'),
        ('theta_world', 'phi_world')
    ]

    for main_axes in svd_axis:
        #svd_features = [f + '_svd' for f in main_axes]
        principal_axis_features = [f + '_contribution_main_axis' for f in main_axes]

        name = '_'.join(main_axes[0].split('_')[:-1])
        svd_ratio_features = [f'{name}_ratio_svd_{i}' for i in range(len(main_axes[1:]))]
        svd_features = [f'{name}_svd_{i}' for i in range(len(main_axes))]

        acc_vec = sequence_data[main_axes].to_numpy()

        window_size = 10
        sv =  np.zeros( (3, len(acc_vec)) )
        sv_ratio = np.zeros( (2, len(acc_vec)) )
        principal_axis = np.zeros( (3, len(acc_vec)) )
        for i in range(window_size, len(acc_vec)):
            window = acc_vec[i-window_size: i]
            U, S, Vt = np.linalg.svd(window - window.mean(axis = 0))
            principal_axis[:, i] =  Vt[0] ** 2
            sv[:, i] = S
            sv_ratio[0, i] = S[1]/S[0]
            sv_ratio[1, i] = S[2]/S[0]

        sequence_data_with_corr_and_svd[svd_features] = sv.T        
        sequence_data_with_corr_and_svd[principal_axis_features] = principal_axis.T
        sequence_data_with_corr_and_svd[svd_ratio_features] = sv_ratio.T 

    phase_transition = sequence_data['phase_adj'] == 0
    phase_gesture = sequence_data['phase_adj'] == 1

    #f_freq = [f for f in sequence_data.columns if ('acc' in f) or ('rotvec' in f) or ('angle_rad' in f) or ('phi' in f) or ('theta' in f) or ('ang_vel' in f)]
    f_freq = [f for f in sequence_data.columns if any(substr in f for substr in ['acc', 'rotvec', 'angle_rad', 'phi', 'theta', 'angle_vel'])]

    for f in f_freq:
        f0_series = np.zeros(len(sequence_data))
        f1_series = np.zeros(len(sequence_data))
        ratio_freq = np.zeros(len(sequence_data))
        if phase_gesture.sum() > 1:  
            extracted_sig = sequence_data.loc[phase_gesture, f]
            f0 = autocorr_frequency(extracted_sig, sampling_rate=10)
            if f0 > 0:
                residual = remove_frequency_component(extracted_sig, f0, sampling_rate=10)
                f1 = autocorr_frequency(residual, sampling_rate=10)
                ratio_freq[phase_gesture] = f1 / f0

            else:
                f1 = 0.
                ratio_freq[phase_gesture] = 0.

            f0_series[phase_gesture] = f0
            f1_series[phase_gesture] = f1
            
        else:
            f0_series[phase_gesture] = 0.
            f1_series[phase_gesture] = 0.
            ratio_freq[phase_gesture] = 0.

        sequence_data_with_corr_and_svd[f'{f}_f0'] = f0_series
        sequence_data_with_corr_and_svd[f'{f}_f1'] = f1_series
        sequence_data_with_corr_and_svd[f'{f}_ratio_freqs'] = ratio_freq


    for sig1, sig2 in corr_features:
        # Initialize correlation series
        corr_series = np.zeros(len(sequence_data))

        if phase_transition.sum() > 1:  
            corr_trans = sequence_data.loc[phase_transition, sig1].corr(sequence_data.loc[phase_transition, sig2])
            corr_series[phase_transition] = corr_trans
        else:
            corr_series[phase_transition] = 0. 

        if phase_gesture.sum() > 1:
            corr_gest = sequence_data.loc[phase_gesture, sig1].corr(sequence_data.loc[phase_gesture, sig2])
            corr_series[phase_gesture] = corr_gest
        else:
            corr_series[phase_gesture] = 0.

        # Save in your dataframe
        sequence_data_with_corr_and_svd[f'{sig1}_{sig2}_corr'] = corr_series

    return sequence_data_with_corr_and_svd

def add_gesture_phase(sequence_data):
    sequence_data_phase = sequence_data.copy()
    length_sequence = len(sequence_data)
    idx_transition = int( 0.45 * length_sequence)
    phase = np.zeros(length_sequence)
    phase[idx_transition:] = 1.
    sequence_data_phase['phase_adj'] = phase
    return sequence_data_phase

def manage_tof(sequence_data, demographics):
    sequence_data_tof = sequence_data.copy()
    #tof_col = []
    for i in range(1, 6):
        pixel_cols = [f for f in sequence_data.columns if f'tof_{i}' in f]
        tof_data = sequence_data[pixel_cols].replace(-1, np.nan)
        sequence_data_tof[f'tof_{i}_mean'] = sequence_data[pixel_cols].mean(axis = 1)
        sequence_data_tof[f'tof_{i}_std'] = sequence_data[pixel_cols].std(axis = 1)
        sequence_data_tof[f'tof_{i}_min'] = tof_data.min(axis = 1)
        sequence_data_tof[f'tof_{i}_max'] = tof_data.max(axis = 1)

    subject = sequence_data['subject'].iloc[0]
    handedness = demographics[demographics['subject'] == subject]['handedness'].iloc[0] ## (0): left, (1): right
    if handedness == 2:
        cols_tof_3 = [col for col in sequence_data.columns if 'tof_3' in col]
        cols_thm_3 = [col for col in sequence_data.columns if 'thm_3' in col]
        cols_tof_5 = [col for col in sequence_data.columns if 'tof_5' in col]
        cols_thm_5 = [col for col in sequence_data.columns if 'thm_5' in col]
        rename_dict = {}
        # TOF3 <-> TOF5
        for c3, c5 in zip(cols_tof_3, cols_tof_5):
            rename_dict[c3] = c5
            rename_dict[c5] = c3

        # THM3 <-> THM5
        for c3, c5 in zip(cols_thm_3, cols_thm_5):
            rename_dict[c3] = c5
            rename_dict[c5] = c3

        sequence_data_tof.rename(columns=rename_dict, inplace=True)

    return sequence_data_tof

# def add_correlations_tof_imu(sequence_data):

def split_into_transition_and_gesture_phases(sequence_data, meta_cols):
    sequence_data_split = sequence_data.copy()
    df_transition = sequence_data[sequence_data['phase_adj'] == 0].drop(columns='phase_adj')
    df_gesture = sequence_data[sequence_data['phase_adj'] == 1].drop(columns='phase_adj')

    # Rename columns
    df_transition = df_transition.add_suffix('_transition')
    df_gesture = df_gesture.add_suffix('_gesture')

    # Pad shorter DataFrame with NaNs to match the longer one
    max_len = max(len(df_transition), len(df_gesture))

    df_transition = df_transition.reset_index(drop=True).reindex(range(max_len))
    df_gesture = df_gesture.reset_index(drop=True).reindex(range(max_len))

    # Concatenate along columns
    df_combined = pd.concat([df_transition, df_gesture], axis=1)
    # Drop transition versions of meta columns
    df_combined.drop(columns=[col + '_transition' for col in meta_cols], inplace=True)
    # Rename gesture versions of meta columns back to original names
    df_combined.rename(columns={col + '_gesture': col for col in meta_cols}, inplace=True)

    sequence_data_split = df_combined.fillna(0)
    return sequence_data_split


def wrapper_data( TRAIN = True, split = False):
    if TRAIN:
        train_df = pd.read_csv(Config.TRAIN_PATH)
        train_demographics = pd.read_csv(Config.TRAIN_DEMOGRAPHICS_PATH)

        label_encoder = LabelEncoder()
        train_df['gesture_id'] = label_encoder.fit_transform(train_df['gesture'].astype(str))
        joblib.dump(label_encoder, os.path.join(Config.EXPORT_DIR, "label_encoder.pkl"))

        gesture_id_to_gestures = {idx: cl for idx, cl in enumerate(label_encoder.classes_)}

        gesture_to_seq_ids = (
            train_df.groupby('gesture_id')['sequence_id']
            .unique()
            .apply(list)
            .to_dict()
        )

        seq_type_to_seq_ids = (
            train_df.groupby('sequence_type')['sequence_id']
            .unique()
            .apply(list)
            .to_dict()
        )

        train_sequence_subject = {
            seq_id: sequence['subject'].iloc[0]
            for seq_id, sequence in train_df.groupby('sequence_id')
        }

        train_sequence_ids = sorted(train_df['sequence_id'].unique())


        train_cols = set(train_df.columns)


        # Group by sequence_id for training data - need to include gesture column for labels
        train_cols = train_cols + ['gesture_id'] if 'gesture_id' not in train_cols else train_cols

        print("Handle quaternion missing values in the train dataset...")
        train_df_clean = clean_and_check_quaternion(train_df[train_cols])


        train_sequences = train_df_clean.groupby('sequence_id')


        split_ids = {
            'classes': gesture_id_to_gestures,
            'train': {
                'train_sequence_ids': train_sequence_ids, ##List of all train ids
                'train_sequence_subject': train_sequence_subject, ##List of all train subject
                'gesture_to_seq_ids': gesture_to_seq_ids, ##dic by gesture
                'seq_type_to_seq_ids': seq_type_to_seq_ids ##dic by sequence_type
            },
        }
        # Save
        with open(os.path.join(Config.EXPORT_DIR, 'split_ids.pkl'), 'wb') as f:
            pickle.dump(split_ids, f)
        

        ### FEATURES ####
        meta_cols = sorted(['gesture', 'gesture_id', 'sequence_type', 'behavior', 'orientation',
                    'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'])
        train_df_clean[meta_cols].to_csv( os.path.join(Config.EXPORT_DIR, 'train_metadata.csv' ))

        features_cols = [c for c in train_cols if c not in meta_cols]
        print("adding new features...")
        processed_sequences = []
        for _, data_sequence in tqdm(train_sequences, desc="Processing Sequences"):
            data_sequence = data_sequence.reset_index(drop=True)
            data_sequence = add_gesture_phase(data_sequence)
            data_sequence = compute_acceleration_features(data_sequence, train_demographics)
            data_sequence = compute_angular_features(data_sequence, train_demographics)
            #data_sequence = compute_fft_features(data_sequence)
            #data_sequence = compute_theta_phi_features(data_sequence) 
            #data_sequence = compute_corr_and_svd_features(data_sequence)
            data_sequence = manage_tof(data_sequence, train_demographics)

            if split:
                data_sequence = split_into_transition_and_gesture_phases(data_sequence, meta_cols)

            #print(data_sequence[['acc_x_transition', 'acc_x_gesture']])

            processed_sequences.append(data_sequence)
    
        train_df_clean = pd.concat(processed_sequences).sort_index()

        train_cols = train_df_clean.columns
        #new_features = [c for c in cols if c not in features_cols and c not in meta_cols]
        features_cols = [c for c in train_cols if c not in meta_cols]
        imu_cols  = sorted([c for c in features_cols if not (c.startswith('thm_') or c.startswith('tof_'))])
        tof_cols  = sorted([c for c in features_cols if c.startswith('tof_')])
        thm_cols  = sorted([c for c in features_cols if c.startswith('thm_')])

        fixed_order_features = np.concatenate( (imu_cols, thm_cols, tof_cols) )


        print(f"all features have been generated")
        # global scaler
        #features_to_exclude = [f for f in fixed_order_features if ('svd' in f) or ('contribution_main_axis' in f) or ('f0' in f)]  # for example
        features_to_exclude = [f for f in fixed_order_features if any(substr in f for substr in ['phase_adj'])]
        features_to_scale = [f for f in fixed_order_features if f not in features_to_exclude]
        print(features_to_scale)
        all_features = np.concatenate( (meta_cols, fixed_order_features) )
        
        for f in train_df_clean.columns:
            if f not in all_features:
                print(f)

        train_df_clean = train_df_clean[all_features]

        scaler = StandardScaler().fit(train_df_clean[features_to_scale].to_numpy())
        joblib.dump(scaler, os.path.join(Config.EXPORT_DIR, "scaler.pkl") )

        train_sequences = train_df_clean.groupby('sequence_id')
        print(train_df_clean.columns)

        cols = {
            #'train': train_cols,
            'meta': meta_cols,
            #'features': features_cols,
            'imu': imu_cols,
            'tof': tof_cols,
            'thm': thm_cols
        }
        with open(os.path.join(Config.EXPORT_DIR, 'cols.pkl'), 'wb') as f:
            pickle.dump(cols, f)


        X, y = build_train_test_data(train_sequences, cols)
        return X, y



def get_info(data_sequences, demograph, seq_id, print_data = False):
    # Filter rows with the given sequence_id
    #seq_id = 'SEQ_051475'
    sequence_data = data_sequences.get_group(seq_id)

    subject_id = sequence_data['subject'].iloc[0]
    subject_demographics = demograph[demograph['subject'] == subject_id]

    seq_info = sequence_data[
        ["sequence_id", "subject", "orientation", "gesture", "gesture_id", "sequence_type"]
    ].head(1).squeeze() 
    demo_info = subject_demographics[
        ["adult_child", "age", "sex", "handedness", 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    ].head(1).squeeze()
    demo_info["adult_child"] = {0: "child", 1: "adult"}.get(demo_info["adult_child"], "unknown")
    demo_info["sex"] = {0: "female", 1: "male"}.get(demo_info["sex"], "unknown")
    demo_info["handedness"] = {0: "left-handed", 1: "right-handed"}.get(demo_info["handedness"], "unknown")
    combined = pd.concat([seq_info, demo_info])
    if print_data:
        display(combined.to_frame(name='Value'))
    return combined

def get_info_v2(demograph, seq_id, seq_id_to_subject, print_data = False):
    subject_id = seq_id_to_subject[seq_id]
    subject_demographics = demograph[demograph['subject'] == subject_id]

    demo_info = subject_demographics[
        ["adult_child", "age", "sex", "handedness", 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    ].head(1).squeeze()
    demo_info["adult_child"] = {0: "child", 1: "adult"}.get(demo_info["adult_child"], "unknown")
    demo_info["sex"] = {0: "female", 1: "male"}.get(demo_info["sex"], "unknown")
    demo_info["handedness"] = {0: "left-handed", 1: "right-handed"}.get(demo_info["handedness"], "unknown")
    if print_data:
        display(demo_info.to_frame(name='Value'))
    return demo_info


def pad_and_truncate(X_batch, maxlen, padding_value=0.0, dtype=torch.float32):
    padded_batch = []
    for seq in X_batch:
        seq = torch.tensor(seq, dtype=dtype)
        length = seq.size(0)

        # Truncate
        if length > maxlen:
            seq = seq[:maxlen]
        # Pad
        elif length < maxlen:
            pad_len = maxlen - length
            padding = torch.full((pad_len, *seq.shape[1:]), padding_value, dtype=dtype)
            seq = torch.cat([seq, padding], dim=0)

        padded_batch.append(seq)

    return torch.stack(padded_batch)  # [batch_size, maxlen, features]

def build_train_test_data(data_sequences, cols, mask_gesture = False):
    X_batch, y_batch, len_seq = [], [], []
    features = np.concatenate( (cols['imu'], cols['thm'], cols['tof']) )
    features_to_exclude = [f for f in features if any(substr in f for substr in ['phase_adj'])]
    features_to_scale = [f for f in features if f not in features_to_exclude]

    idx_to_scale = np.where(np.isin(features, features_to_scale))[0]
    #idx_to_exclude = np.where(np.isin(features, features_to_exclude))[0]

    seq_ids = []
    for seq_id, data_sequence in data_sequences:
        if mask_gesture:
            gesture_phase = data_sequence['phase'] == 'Gesture'
            sequence = data_sequence[features][gesture_phase]
        else:
            sequence = data_sequence[features]
        
        sequence = sequence.to_numpy()

        # Fit and transform only those columns
        scaler = joblib.load( os.path.join(Config.EXPORT_DIR, "scaler.pkl") )
        if len(sequence) > 0:
            sequence[:, idx_to_scale] =  scaler.transform(sequence[:, idx_to_scale])

        #print(sequence[['linear_acc_ratio_svd_0', 'linear_acc_ratio_svd_1']])
        #cols_to_scale = [c for c in cols['imu'] if c.startswith('acc_')]
        #sequence[cols_to_scale] = scaler.fit_transform(sequence[cols_to_scale])

        X_batch.append(sequence)
        seq_ids.append(seq_id)
        y_batch.append(data_sequence['gesture_id'].iloc[0])
        len_seq.append(len(sequence))

    ### labels one-hot categorical ###
    y_final = torch.tensor(y_batch)
    #y_final = F.one_hot(y_torch, num_classes = num_classes).float()
    
    ### pad and truncate sequences to the 95 percentile
    pad_len_seq = int(np.percentile(len_seq, Config.PERCENTILE))
    X_final = pad_and_truncate(X_batch, maxlen=pad_len_seq)

    return X_final, y_final #, seq_ids


### COMPETITION METRIC ###

def competition_metric(y_true, y_pred) -> tuple:
    """Calculate the competition metric (Binary F1 + Macro F1) / 2"""
    BFRB_gesture = [0, 1, 3, 4, 6, 7, 9, 10]
    #non_BFRB_gesture = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17]
     
    # Binary F1: BFRB vs non-BFRB
    binary_f1 = f1_score(
        np.where(np.isin(y_true, BFRB_gesture), 1, 0),
        np.where(np.isin(y_pred, BFRB_gesture), 1, 0),
        zero_division=0.0,
    )

    binary_recall =  recall_score(
        np.where(np.isin(y_true, BFRB_gesture), 1, 0),
        np.where(np.isin(y_pred, BFRB_gesture), 1, 0),
        zero_division=0.0,
    )
    
    # Macro F1: specific gesture classification (only for BFRB gestures)
    macro_f1 = f1_score(
        np.where(np.isin(y_true, BFRB_gesture), y_true, 99),  # Map non-BFRB to 99
        np.where(np.isin(y_pred, BFRB_gesture), y_pred, 99),  # Map non-BFRB to 99
        average="macro", 
        zero_division=0.0,
    )
    
    # Final competition score
    final_score = 0.5 * (binary_f1 + macro_f1)
    
    return final_score, binary_recall, macro_f1

def reset_seed(seed=42):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#### TO STORE TRIALS #####

def save_hparams_and_architecture(hparams: dict, model: torch.nn.Module, file_txt_path: str):
    with open(file_txt_path, 'w') as f:

        f.write("======== COMMENTS ========\n")
        if not hparams['C_TOF_RAW']:
            f.write("C = 1 for weights TOF RAW (same weights accross all TOF)\n")
        else:
            f.write("C = 5 for weights TOF RAW (diff weights accross all TOF)\n")
        f.write(f"Normalization with mask 1 for TOF RAW = {hparams['normalisation_TOF_RAW']}\n")
        f.write(f"Normalization with mask 1 for TOF-THM = {hparams['normalisation_TOF_THM']}\n")
        f.write(f"Fused attention gate = {hparams['attention_for_fusion']}\n")
        f.write(f"Pooled attention gate = {hparams['attention_pooled']}\n")

        f.write("===== Hyperparameters =====\n")
        for k, v in hparams.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\n===== Model Architecture =====\n")
        f.write(str(model))

        f.write("\n\n===== Forward Method =====\n")
        f.write(inspect.getsource(model.forward))


def log_best_scores(scores_dict: dict, file_txt_path: str):
    """
    scores_dict format:
    {
        'mixture': [0.75, 0.78, 0.72, 0.74, 0.77],
        'imu_only': [...],
        'imu_tof_thm': [...],
    }
    """
    with open(file_txt_path, 'a') as f:
        f.write("\n===== Best Scores per Fold =====\n")
        for key, scores in scores_dict.items():
            best = np.mean(scores)
            f.write(f"Mean-fold {key} score: {best:.4f} (Fold scores: {scores})\n")

def save_hparams_and_architecture_to_pkl(hparams: dict, model: torch.nn.Module, file_pkl_path: str):
    data_to_save = {
        'hyperparams': hparams,
        'model_architecture': str(model)
    }
    with open(file_pkl_path, 'wb') as f:
        pickle.dump(data_to_save, f)

def bytes_to_gb(x):
    return round(x / (1024**3), 2)

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Memory usage:")
    print(f"  Total     : {bytes_to_gb(mem.total)} GB")
    print(f"  Used      : {bytes_to_gb(mem.used)} GB")
    print(f"  Available : {bytes_to_gb(mem.available)} GB")
    print(f"  Percent   : {mem.percent}%")
