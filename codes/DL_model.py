import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from collections import Counter

from modules.functions import *
from modules.class_models import *
from modules.training_functions import *
from modules.competition_metric import CompetitionMetric

import sys


print("")
print("======== COMMENTS ========")
print(" normal loss and BRFB loss only, gamma = 0.7")
print("+ balanced class weights for random sampling but not for loss function")
print("-------------------------")
print("")

N_SPLITS = 5
BATCH_SIZE = 64
EPOCHS = 250
PATIENCE = 50
ALPHA = 0.3
LR = 1e-3

SEED = Config.SEED
reset_seed(SEED)

file_path_train = os.path.join(Config.EXPORT_DIR, "train_torch_tensors_from_wrapper_not_split.pt")
file_path_cols = os.path.join(Config.EXPORT_DIR, "cols.pkl")
file_path_splits = os.path.join(Config.EXPORT_DIR, "split_ids.pkl")


selected_features = sys.argv[1:]

if len(selected_features) == 0:
    selected_features = [
        'acc_x','acc_y','acc_z',#,'rot_x', 'rot_y', 'rot_z', 'rot_w', 
        'rotvec_x', 'rotvec_y', 'rotvec_z', 
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 
        #'linear_acc_x_FFT', 'linear_acc_y_FFT', 'linear_acc_z_FFT', 
        #'acc_norm_world', 
        # 'acc_norm', 'linear_acc_norm', 
        # 'acc_norm_jerk', 'linear_acc_norm_jerk', 
        #'angle_rad', 'angular_speed', 
        # 'rot_angle', 'rot_angle_vel', 'angular_speed', 
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'ang_dist',
        # 'ang_vel_x_FFT', 'ang_vel_y_FFT', 'ang_vel_z_FFT', 
        'phase_adj',
        ] 

print("Features:", selected_features)


# ---------------- LOAD DATA ------------------------


if os.path.exists(file_path_train):
    print("Loading existing tensor...")
    data = torch.load(file_path_train)
    X_train, y_train = data['X_train'], data['y_train']

    with open(file_path_cols, 'rb') as f:
        cols = pickle.load(f)

    with open(file_path_splits, 'rb') as f:
        split_ids = pickle.load(f)


else:
    print("File not found. Generating data...")
    X_train, y_train = wrapper_data(split=False)
    print(X_train.shape, y_train.shape)

    data = {'X_train': X_train, 'y_train': y_train} 
    torch.save(data, file_path_train)
    print("Data saved.")

    with open(file_path_cols, 'rb') as f:
        cols = pickle.load(f)

    with open(file_path_splits, 'rb') as f:
        split_ids = pickle.load(f)


gesture_mapping = {cl: idx for idx, cl in split_ids['classes'].items()}   ### GESTURE MAP CLASSES --> LABELS
bfrb_gesture = CompetitionMetric().target_gestures                        ### TARGET GESTURE CLASSES
bfrb_classes = torch.tensor([gesture_mapping[cl] for cl in bfrb_gesture]) ### TARGET GESTURE LABELS


# ------------------ SELECT FEATURES AND PREPARE DATA FOR TRAINING ------------------------

all_features = np.concatenate( (cols['imu'], cols['thm'], cols['tof']) ) 


idx_imu = [np.where(all_features == f)[0][0] for f in selected_features]    ### select features from selected_features above
idx_tof = np.where(np.isin(all_features, cols['tof']))[0]                   ### TOF Features for later
idx_thm = np.where(np.isin(all_features, cols['thm']))[0]                   ### THM Features for later

X = X_train[:, :, idx_imu]   ## select idx features in X
y = y_train                  ## labels 


#### NaN ? in DATA #### 
nan_mask = torch.isnan(X)
nan_indices = torch.nonzero(nan_mask, as_tuple=True)
print(f"number of NaN (detect possible FE errors): {len(np.unique(nan_indices[0].numpy()))}")
      
if len(np.unique(nan_indices[0].numpy())) > 0:      
    X = torch.tensor(np.nan_to_num(X, nan=0.0))
########################

print(f"Data shape (X, y): {X.shape, y.shape}")

# cw_vals = compute_class_weight('balanced', classes=list(split_ids['classes'].keys()), y=y.numpy())  ## Class weights to handle imbalance
# class_weight = torch.from_numpy(cw_vals).float()                                                    ## class weights as torch tensor

class_weight = 0.7 * torch.ones(len(split_ids['classes'].keys()))
class_weight[bfrb_classes] = 2.

# ------------------------------- TRAINING ---------------------------------

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state = 39) #STRATIFIED k-Fold by group (subject_id)

train_ids = np.array(split_ids['train']['train_sequence_ids']) #seq_id of data sequences 
groups = [split_ids['train']['train_sequence_subject'][seq_id] for seq_id in train_ids] #subject_id of data_sequences

# idx_spe_seq = np.where(train_ids == 'SEQ_000007')[0]


### LOOP FOR EACH TRAINING FOLD
best_scores = []
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    print(f"\n===== FOLD {fold+1}/{N_SPLITS} =====\n")
    reset_seed(SEED)

    # Split data
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]


    subjects_id = np.array(groups)[train_idx]
    train_seq_ids = train_ids[train_idx]

    print(" ---- check for reproductibility ----")
    print(f"first 10 seq_id = {train_seq_ids[:10]}")
    print(f"first 10 train idx = {train_idx[:10]}, and val idx = {val_idx[:10]}")
    print(f"mean train idx = {np.mean(train_idx)}, and mean val idx = {np.mean(val_idx)}\n")

    df = pd.DataFrame({'subject_id': subjects_id, 'seq_id': train_seq_ids})
    seqs_by_subject = (
            df.groupby('subject_id')['seq_id']
            .unique()
            .apply(list)
            .to_dict()
        )

    #### DATA AUGMENTATION #####
    print("------ DATA AUGMENTATION: DEVICE ROTATION ------")
    rotation_augmented = DeviceRotationAugment(X_tr, y_tr, train_seq_ids,     
                          seqs_by_subject, selected_features, p_rotation=1.1, small_rotation=2)
    X_tr, y_tr, count = rotation_augmented(axes=['z', 'x'])
    print(f"number of additional rotated features samples: {count}")
    print(f"shape of training data after augmentation (X, y): {X_tr.shape, y_tr.shape}\n")

    #augmenter = Augment()

    # augmenter = Augment(
    #     p_jitter=0.98, sigma=0.033, scale_range=(0.75,1.16),
    #     p_dropout=0.42,
    #     p_moda=0.39, drift_std=0.004, drift_max=0.39    
    # )

    #########################################

    train_ds = SensorDataset(X_tr, y_tr, imu_dim = 7, alpha=ALPHA)  ### TRAINING ROTATION AUGMENTED DATA WITH MixUp \alpha 
    val_ds = SensorDataset(X_val, y_val, imu_dim = 7, training=False) ### VALIDATION DATA (NO AUG, NO MixUp)


    # CLASS IMBALANCE handling 
    print(" ----------- CLASS INBALANCE SAMPLER (WeightedRandomSampler) ---------") 
    class_counts = np.bincount(y_tr.numpy())
    print(f"Number of samples per class: {Counter(y_tr.numpy())}\n")
    class_weights_balanced = 1. / class_counts
    sample_weights = class_weights_balanced[y_tr.numpy()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights) , replacement=True)
    tracking_sampler = TrackingSampler(sampler)

    sampled_indices = list(sampler)
    sampled_labels = y_tr[sampled_indices]
    print(Counter(sampled_labels.numpy()))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=tracking_sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = soft_cross_entropy # LOSS FUNCTION

    model = MiniGestureClassifier(imu_dim=X_tr.shape[2], hidden_dim=128, num_classes=len(class_weight)) # MODEL
    optimizer = optim.Adam(model.parameters(), lr=LR) # OPTIMIZER

    best_score = train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, DEVICE, class_weight, bfrb_classes, patience=PATIENCE, fold = fold)
    best_scores.append(best_score)


for fold, score in enumerate(best_scores):
    print(f" - Best score for fold {fold}: {score}")

print(f"mean score for all folds: {np.mean(best_scores)}")

