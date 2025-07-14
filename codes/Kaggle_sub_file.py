import polars as pl
import os, joblib
import torch
import pickle
import torch
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score,  recall_score
from tqdm import tqdm
from collections import Counter
#import kaggle_evaluation.cmi_inference_server

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config: 
    """Central configuration class for training and data parameters"""
    
    # Paths for Kaggle environment
    TRAIN_PATH = "~/.kaggle/sensor-data/train.csv"
    TRAIN_DEMOGRAPHICS_PATH = "~/.kaggle/sensor-data/train_demographics.csv"
    TEST_PATH = "~/.kaggle/sensor-data/test.csv"
    TEST_DEMOGRAPHICS_PATH = "~/.kaggle/sensor-data/test_demographics.csv"
    EXPORT_DIR =  "/Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/data"                                  
    EXPORT_MODELS_PATH =  "/Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/models"  

    os.makedirs(EXPORT_DIR, exist_ok=True)                                 
    os.makedirs(EXPORT_MODELS_PATH, exist_ok=True)                                 
    
    # Training parameters
    # SEED = 42
    # N_SPLITS = 5
    # BATCH_SIZE = 64
    # EPOCHS = 250
    # PATIENCE = 50
    # ALPHA = 0.3
    # LR = 1e-3

    PADDING = 127

# SEED = Config.SEED
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)     



# =============================================================================
# FUNCTIONS FOR FEATURES ENGINEERING AND CLEAN UP DATA
# =============================================================================

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


def compute_acceleration_features(sequence_data):
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

    return sequence_data_with_acc

def compute_angular_features(sequence_data, time_delta = 10):
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

    return sequence_data_with_ang_vel

def add_gesture_phase(sequence_data):
    sequence_data_phase = sequence_data.copy()
    length_sequence = len(sequence_data)
    idx_transition = int( 0.45 * length_sequence)
    phase = np.zeros(length_sequence)
    phase[idx_transition:] = 1.
    sequence_data_phase['phase_adj'] = phase
    return sequence_data_phase

def manage_tof(sequence_data):
    sequence_data_tof = sequence_data.copy()
    #tof_col = []
    for i in range(1, 6):
        pixel_cols = [f for f in sequence_data.columns if f'tof_{i}' in f]
        tof_data = sequence_data[pixel_cols].replace(-1, np.nan)
        sequence_data_tof[f'tof_{i}_mean'] = sequence_data[pixel_cols].mean(axis = 1)
        sequence_data_tof[f'tof_{i}_std'] = sequence_data[pixel_cols].std(axis = 1)
        sequence_data_tof[f'tof_{i}_min'] = tof_data.min(axis = 1)
        sequence_data_tof[f'tof_{i}_max'] = tof_data.max(axis = 1)
    return sequence_data_tof

def check_gpu_availability():

    import torch
    if torch.backends.mps.is_available():
        print("CUDA is available.")
        return 'mps'
    else:
        print("CUDA not available. Using CPU.")
        return 'cpu'

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

# =============================================================================
# CLASSES (PREDICTOR, )
# =============================================================================

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, bias_strength = 5.):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )
        self.bias_strength = bias_strength
        self.weights = None

    def forward(self, x, phase_adj = None):
        # x: [B, hidden_dim, T]
        scores = self.attn(x).squeeze(1)  # [B, T]

        if phase_adj is not None:
            #bias = (phase_adj.float() * self.bias_strength)  # [B, T]
            scores = scores #+ bias

        weights = F.softmax(scores, dim=1)  # [B, T]
        self.weights = weights
        pooled = torch.sum(x * weights.unsqueeze(1), dim=2)  # [B, hidden_dim]
        return pooled

class IMUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(), #inplace=True
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.3), 
            #nn.MaxPool1d(kernel_size=2, stride=2),  # halves time length
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            #nn.MaxPool1d(kernel_size=2, stride=2),   # halves again → total /4
        )

    def forward(self, x):
        # x: [B, T, input_dim] → [B, input_dim, T]
        x = x.permute(0, 2, 1)
        out = self.net(x)  # [B, hidden_dim, T]
        return out


class MiniGestureClassifier(nn.Module):
    def __init__(self, imu_dim, hidden_dim, num_classes):
        super().__init__()

        self.imu_encoder = IMUEncoder(imu_dim, hidden_dim)
        self.attn_pool = AttentionPooling(hidden_dim)
    

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim, num_classes),
            #nn.Softmax()
        )

    def forward(self, imu, return_attention=False): #, phase_adj = None,
        B, T, _ = imu.shape

        imu_feat = self.imu_encoder(imu)  # [B, hidden_dim, T]

        pooled = imu_feat.mean(dim=2)#   # [B, hidden_dim]
        #pooled = self.attn_pool(imu_feat)

        out = self.classifier_head(pooled)  # [B, num_classes]

        return (out, self.attn_pool.weights) if return_attention else out


class EnsemblePredictor:
    def __init__(self,  processing_dir, models_dir, device):
        self.device = device
        self.models = []
        self.scaler = None
        self.features = None
        self.label_encoder = None
        self.map_classes = None
        self.inverse_map_classes = None
        self.cols = None
        self._load_models(models_dir)
        self._load_processing(processing_dir)

    def _load_models(self, models_dir):
        model_files = sorted(glob.glob(f"{models_dir}/best_model_fold_*.pth"))
        print(f"{len(model_files)} models have been found")
        
        for model_file in model_files:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
            
            model = MiniGestureClassifier(imu_dim=14, hidden_dim=128, num_classes=18)
            
            model.load_state_dict(checkpoint) #['model_state_dict']
            model.to(self.device)
            model.eval()
            self.models.append(model)
            
            print(f"✅ {model_file} loaded")

    def _load_processing(self, processing_dir):
        self.scaler = joblib.load(os.path.join(processing_dir, "scaler.pkl"))
        self.label_encoder = joblib.load(os.path.join(processing_dir, "label_encoder.pkl"))
        self.map_classes = {idx: cl for idx, cl in enumerate(self.label_encoder.classes_)}
        self.inverse_map_classes = {cl: idx for idx, cl in enumerate(self.label_encoder.classes_)}

        
        file_path_cols = os.path.join(processing_dir, "cols.pkl")
        with open(file_path_cols, 'rb') as f:
            self.cols = pickle.load(f)
        self.features = np.concatenate( (self.cols['imu'], self.cols['thm'], self.cols['tof']) ) 

        print("-> scaler, features, labels classes loaded")
        #print(f"features = {self.features}")
    
    def features_eng(self, df_seq: pd.DataFrame):
        df_seq = regularize_quaternions_per_sequence(df_seq)

        ### -- ADD NEW FEATURES (IMU + AVERAGED TOF COLUMNS) --- 
        df_seq = df_seq.reset_index(drop=True)
        df_seq = add_gesture_phase(df_seq)
        df_seq = compute_acceleration_features(df_seq)
        df_seq = compute_angular_features(df_seq)
        df_seq = manage_tof(df_seq)
        return df_seq
    
    def scale_pad_and_transform_to_torch_sequence(self, df_seq, pad_length, is_imu_only = True):
        ### -- Columns re-ordering to match train order
        df_seq_features = df_seq[self.features].copy()

        #has_nan_tof_thm = df_seq_features[ np.concatenate( (self.cols['tof'], self.cols['thm']) ) ].isnull().all(axis=1).all()
        # if has_nan_tof_thm:
        #     print("NaN values have been found in TOF and/or THM data")
        
        has_nan_imu = df_seq_features[self.cols['imu']].isnull().any().any()
        if has_nan_imu:
            print("x IMU cols have NaN values. Shouldn't be the case! Check data!")

        ### -- Scale features and check NaN for IMU COLS  
        np_seq_features  =  df_seq_features.to_numpy()
        features_to_exclude = [f for f in self.features if any(substr in f for substr in ['phase_adj'])]
        features_to_scale = [f for f in self.features if f not in features_to_exclude]
        idx_to_scale = np.where(np.isin(self.features, features_to_scale))[0]
        if len(np_seq_features) > 0:
            np_seq_features[:, idx_to_scale] =  self.scaler.transform(np_seq_features[:, idx_to_scale])


        if is_imu_only:
            imu_features = [
            'acc_x','acc_y','acc_z', 'rotvec_x', 'rotvec_y', 'rotvec_z', 
            'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'ang_dist',
            'phase_adj'
            ] 
            idx_imu = [np.where(self.features == f)[0][0] for f in imu_features]    ### select features from selected_features above
            np_seq_features = np_seq_features[:, idx_imu]


        seq = torch.tensor(np_seq_features, dtype=torch.float32)
        length = seq.size(0)
        # Truncate
        if length >= pad_length:
            seq = seq[:pad_length].unsqueeze(0)
        # Pad
        elif length < pad_length:
            pad_len = pad_length - length
            padding = torch.full((pad_len, *seq.shape[1:]), 0.0, dtype=torch.float32)
            seq = torch.cat([seq, padding], dim=0).unsqueeze(0)

        #print(f"sequence has been scaled and padded. shape (1, T, F): {seq.shape}")
        return seq.to(self.device)
    
    def predict(self, torch_seq):
        pred_by_model = []
        for model in self.models:
        #model = predictor.models[0]
            with torch.no_grad():
                output = model(torch_seq)
                pred = output.argmax(1).cpu().numpy()[0]
        # if seq_id == 'SEQ_000007':
        #     print(pred)
        #     break
                pred_by_model.append(pred)
        most_common_prediction = Counter(pred_by_model).most_common(1)[0][0]
        return self.map_classes[most_common_prediction]


# =============================================================================
# MAIN CODE
# =============================================================================
pad_length = Config.PADDING
processing_dir = Config.EXPORT_DIR
models_dir = Config.EXPORT_MODELS_PATH
test_path = Config.TEST_PATH
train_path = Config.TRAIN_PATH
train_path_demo = Config.TRAIN_DEMOGRAPHICS_PATH

# Check GPU availability
DEVICE = torch.device(check_gpu_availability())
print(f"✓ Configuration loaded for Kaggle environment (Device: {DEVICE})")


predictor = EnsemblePredictor(processing_dir, models_dir, DEVICE)
inverse_map_classes = predictor.inverse_map_classes
map_classes = predictor.map_classes

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    sequence = sequence.to_pandas()
    sequence = predictor.features_eng(sequence)
    torch_seq = predictor.scale_pad_and_transform_to_torch_sequence(sequence, pad_length)
    most_common_prediction = predictor.predict(torch_seq)
    return str(most_common_prediction)


train = pd.read_csv(train_path)
train_demo = pd.read_csv(train_path_demo)

print(f"---> Original shape = {train.shape}")
sel_seq  = train["sequence_id"].unique()#[0 : 3500]
seq      = sel_seq[0: 1750]
oth_cols = sorted([c for c in train.columns if (c.startswith('thm_') or c.startswith('tof_'))]) #train.columns[16:]
train    = train.loc[train.sequence_id.isin(sel_seq)]
train.loc[train.sequence_id.isin(seq), oth_cols] = np.nan
print(f"---> Truncated shape = {train.shape}")
train_sequences = train.groupby("sequence_id")

# sel_seq  = train["sequence_id"].unique()[0 : 10]
# train    = train.loc[train.sequence_id.isin(sel_seq)]


ypred = []
ytruth = []
for _, sequence in tqdm(train_sequences, desc="Processing Sequences"):
#     #print(f"======== SEQUENCE {seq_id} ========")
    sequence = pl.DataFrame(sequence)
    pred = predict(sequence, train_demo)
    ypred.append(inverse_map_classes[pred])
    sequence = sequence.to_pandas()
    ytruth.append(inverse_map_classes[sequence['gesture'].iloc[0]])


print(competition_metric(ytruth, ypred))




