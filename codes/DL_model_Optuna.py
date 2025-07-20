import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedGroupKFold
#from sklearn.utils.class_weight import compute_class_weight


from modules.functions import *
from modules.class_models import *
from modules.training_functions import *
from modules.competition_metric import CompetitionMetric

import sys
import optuna
import logging

print("")
print("======== COMMENTS ========")
print("OPTUNA DEVICE ROTATION HYPERPARAMETERS")
#print("+ balanced class weights for random sampling but not for loss function")
print("-------------------------")
print("")

N_SPLITS = 5
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 50
ALPHA = 0.3
LR = 1e-3
TRAIN = True

SEED = Config.SEED
reset_seed(SEED)

file_path_train = os.path.join(Config.EXPORT_DIR, "train_torch_tensors_from_wrapper_not_split.pt")
file_path_cols = os.path.join(Config.EXPORT_DIR, "cols.pkl")
file_path_splits = os.path.join(Config.EXPORT_DIR, "split_ids.pkl")


selected_features = [] 
EXP = sys.argv[1]
print(f"------------- OPTUNA EXP {EXP} -------------")
print("")

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


train_ids = np.array(split_ids['train']['train_sequence_ids']) #seq_id of data sequences 
groups = [split_ids['train']['train_sequence_subject'][seq_id] for seq_id in train_ids] #subject_id of data_sequences

# ------------------------------- TRAINING ---------------------------------
def objective(trial):

    trial_id = trial.number
    log_file = os.path.join(Config.OPTUNA_PATH_LOGS, f'optuna_exp{EXP}', f"trial_{trial_id}.log")
    
    logger = logging.getLogger(f"trial_{trial_id}")
    handler = logging.FileHandler(log_file)
    #handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Starting trial {trial_id}")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state = 39) #STRATIFIED k-Fold by group (subject_id)

    p_rotation = trial.suggest_uniform("p_rotation", 0.0, 1.)
    small_rotation = trial.suggest_uniform("small_rotation", 0, 5)
    max_x_rotation = trial.suggest_uniform("max_x_rotation", 0, 60)

    ### LOOP FOR EACH TRAINING FOLD
    best_scores = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"\n===== FOLD {fold+1}/{N_SPLITS} =====\n")
        reset_seed(SEED)

        # Split data
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if TRAIN:

            subjects_id = np.array(groups)[train_idx]
            train_seq_ids = train_ids[train_idx]

            logger.info(" ---- check for reproductibility ----")
            logger.info(f"first 10 seq_id = {train_seq_ids[:10]}")
            logger.info(f"first 10 train idx = {train_idx[:10]}, and val idx = {val_idx[:10]}")
            logger.info(f"mean train idx = {np.mean(train_idx)}, and mean val idx = {np.mean(val_idx)}\n")

            df = pd.DataFrame({'subject_id': subjects_id, 'seq_id': train_seq_ids})
            seqs_by_subject = (
                    df.groupby('subject_id')['seq_id']
                    .unique()
                    .apply(list)
                    .to_dict()
                )

            #### DATA AUGMENTATION #####
            logger.info("------ DATA AUGMENTATION: DEVICE ROTATION ------")
            rotation_augmented = DeviceRotationAugment(X_tr, y_tr, train_seq_ids,     
                                seqs_by_subject, selected_features, p_rotation=p_rotation, small_rotation=small_rotation, x_rot_range = (0, max_x_rotation))
            X_tr, y_tr, count = rotation_augmented(axes=['z', 'x'])
            logger.info(f"number of additional rotated features samples: {count}")
            logger.info(f"shape of training data after augmentation (X, y): {X_tr.shape, y_tr.shape}\n")

            #augmenter = Augment()

            # augmenter = Augment(
            #     p_jitter=0.98, sigma=0.033, scale_range=(0.75,1.16),
            #     p_dropout=0.42,
            #     p_moda=0.39, drift_std=0.004, drift_max=0.39    
            # )

            #########################################

            train_ds = SensorDataset(X_tr, y_tr, imu_dim = 7, alpha=ALPHA)  ### TRAINING ROTATION AUGMENTED DATA WITH MixUp \alpha 


            # CLASS IMBALANCE handling 
            logger.info(" ----------- CLASS INBALANCE SAMPLER (WeightedRandomSampler) ---------") 
            class_counts = np.bincount(y_tr.numpy())
            logger.info(f"Number of samples per class: {Counter(y_tr.numpy())}\n")
            class_weights_balanced = 1. / class_counts
            sample_weights = class_weights_balanced[y_tr.numpy()]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights) , replacement=True)
            tracking_sampler = TrackingSampler(sampler)

            sampled_indices = list(sampler)
            sampled_labels = y_tr[sampled_indices]
            logger.info(Counter(sampled_labels.numpy()))

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=tracking_sampler)
        
            val_ds = SensorDataset(X_val, y_val, imu_dim = 7, training=False) ### VALIDATION DATA (NO AUG, NO MixUp)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


        if TRAIN:
            criterion = SoftCrossEntropy() # LOSS FUNCTION #bfrb_classes=bfrb_classes, gamma = .5, lamb = .5

            reset_seed(SEED)
            model = MiniGestureClassifier(imu_dim=X_tr.shape[2], hidden_dim=128, num_classes=len(class_weight)) # MODEL
            optimizer = optim.Adam(model.parameters(), lr=LR) # OPTIMIZER

            best_score = train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, BATCH_SIZE, DEVICE, bfrb_classes, patience=PATIENCE, fold = fold, logger = logger)
            best_scores.append(best_score)
        else:
            print("---- INFERENCE MODE ----")
            processing_dir = Config.EXPORT_DIR
            models_dir = Config.EXPORT_MODELS_PATH
            predictor = EnsemblePredictor(processing_dir, models_dir, DEVICE)
            inverse_map_classes = predictor.inverse_map_classes
            #map_classes = predictor.map_classes
            
            preds_str = predictor.predict(X_val.to(DEVICE), by_fold = fold)
            preds_int = [inverse_map_classes[pred_str] for pred_str in preds_str]
            best_score, _, _ = competition_metric(y_val, preds_int)
        
        best_scores.append(best_score)

    logger.info(f"Finished trial {trial_id}")
    return np.mean(best_scores)

study_name = "device_rotation_hyperparameters"
study = optuna.create_study(direction="maximize",study_name=study_name,)
study.optimize(objective, n_trials=50, n_jobs=-1)

print("Best hyperparameters:", study.best_params)
print("Best validation score:", study.best_value)

# List all tried hyperparameters
print("\n All tried hyperparameter sets:")
for trial in study.trials:
    print(f"Trial {trial.number}: {trial.params} → val_score = {trial.value:.4f}")

### SAVE OPTUNA RESULTS IN A DIC
optuna_params = {'tried_params': {trial.number: {} for trial in study.trials}}
optuna_params['best_params'] = study.best_params | {'score': study.best_value}
for trial in study.trials:
    optuna_params['tried_params'][trial.number] = trial.params | {'score': trial.value}

with open(os.path.join(Config.OPTUNA_PATH_SAVED, study_name + '.pkl'), 'wb') as f:
    pickle.dump(optuna_params, f)

# with open(os.path.join(Config.OPTUNA_PATH_SAVED, study_name + '.pkl'), 'rb') as f:
#         optuna_params = pickle.load(f)

# print(optuna_params)