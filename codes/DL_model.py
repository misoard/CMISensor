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
from datetime import datetime


file_path_trial = os.path.join(Config.EXPORT_DIR, 'number_trials.txt')
with open(file_path_trial, 'r') as f:
    trial = f.read()

last_trial = int(trial)
current_trial = last_trial + 1

with open(file_path_trial, 'w') as f:
    f.write(str(current_trial))

TRAIN = True
N_TRIAL = current_trial

print(f"=============== TRIAL {N_TRIAL} ===============")

data_file =  "train_torch_tensors_from_wrapper_left_corrected_without_TOF_correction.pt"

N_SPLITS = 5
BATCH_SIZE = 64
EPOCHS = 160
HIDDEN_DIM = 128
PATIENCE = 45
ALPHA = 0.4
LR = 1e-3
SEED_CV_FOLD = 39

p_dropout = 0.48 #0.42
p_jitter= 0.2 #0.98
p_moda = 0.2 #0.4
p_rotation = 1.1
small_rotation = 2.
x_max_angle = 30.
y_max_angle = 15.
axes_rotation = ['z', 'x', 'y']

normalisation_TOF_RAW = False
normalisation_TOF_THM = True
attention_for_fusion = False
attention_pooled = True
C_TOF_RAW = False
ADD_TOF_TO_THM = True

SCHEDULER = True
patience_scheduler = 8
factor_scheduler = 0.7


GAMMA = 0.0
LAMB = 0.0
L_IMU = 0.25


SEED = Config.SEED
reset_seed(SEED)


file_path_train = os.path.join(Config.EXPORT_DIR, data_file)
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
selected_tof = [f for f in cols['tof'] if ('v' not in f) and ('tof_5' not in f)]
raw_tof = [f for f in cols['tof'] if ('v' in f) and ('tof_5' not in f)]
print(raw_tof)

raw_tof_sorted = np.array([f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)])
check_all_pixels = np.array([f in raw_tof for f in raw_tof_sorted]   )            ### THM Features for later

if not np.all(check_all_pixels):
    print(f"missing pixel raw data in TOF data: {np.array(raw_tof_sorted)[~check_all_pixels]}")



raw_tof_sorted = list(raw_tof_sorted[check_all_pixels])


idx_imu = [np.where(all_features == f)[0][0] for f in selected_features]    ### select features from selected_features above
idx_tof = [np.where(all_features == f)[0][0] for f in selected_tof]                   ### TOF Features for later
idx_raw_tof = [np.where(all_features == f)[0][0] for f in raw_tof_sorted]                   ### TOF Features for later
idx_thm = [np.where(all_features == f)[0][0] for f in cols['thm'] if 'thm_5' not in f]               ### THM Features for later


idx_all = idx_imu + idx_thm + idx_tof + idx_raw_tof
indices_branches = {
    'imu': np.arange(len(idx_imu)), 
    'thm': np.arange(len(idx_imu), len(idx_imu + idx_thm)), 
    'tof': np.arange(len(idx_imu + idx_thm), len(idx_imu + idx_thm + idx_tof)),
    'tof_raw': np.arange(len(idx_imu + idx_thm + idx_tof), len(idx_all))
    }
# else:
#     idx_all = idx_imu + idx_thm + idx_raw_tof
#     indices_branches = {
#         'imu': np.arange(len(idx_imu)), 
#         'thm': np.arange(len(idx_imu), len(idx_imu + idx_thm)), 
#         'tof': [],
#         'tof_raw': np.arange(len(idx_imu + idx_thm), len(idx_all))
#         }


X = X_train[:, :, idx_all]   ## select idx features in X
y = y_train                  ## labels 


#X[:, :, indices_branches['thm'][0]:] = float('nan')
# for sample in range(len(X)):
#     if X[sample, :, indices_branches['imu']].shape[1] == 0:
#         print("houla")

#### NaN ? in DATA #### 
nan_mask = torch.isnan(X[:, :, indices_branches['imu']])
nan_indices = torch.nonzero(nan_mask, as_tuple=True)
print(f"number of NaN (detect possible FE errors): {len(np.unique(nan_indices[0].numpy()))}")
      
if len(np.unique(nan_indices[0].numpy())) > 0:      
    X[:, :, indices_branches['imu']] = torch.tensor(np.nan_to_num(X[:, :, indices_branches['imu']], nan=0.0))

nan_mask = torch.isnan(X)
nan_indices = torch.nonzero(nan_mask, as_tuple=True)
      
if len(np.unique(nan_indices[0].numpy())) > 0:      
    X = torch.tensor(np.nan_to_num(X, nan=0.0))

########################


print(f"Data shape (X, y): {X.shape, y.shape}")

# cw_vals = compute_class_weight('balanced', classes=list(split_ids['classes'].keys()), y=y.numpy())  ## Class weights to handle imbalance
# class_weight = torch.from_numpy(cw_vals).float()                                                    ## class weights as torch tensor

class_weight = 0.7 * torch.ones(len(split_ids['classes'].keys()))
class_weight[bfrb_classes] = 2.


#### ---------------- PARAMETERS --------------------

# device_rotation_params = {'p_rotation': 0.7497695732931614, 'small_rotation': 1.7353531580176353, 'max_x_rotation': 24.144699565481698}

# ----------- ALL PARAMETERS TO SAVE IT ---------------
all_parameters = {
    "data_file": data_file,
    "SEED": SEED,
    "SEED_CV_FOLD": SEED_CV_FOLD if SEED_CV_FOLD is not None else None,
    "N_SPLITS": N_SPLITS,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "HIDDEN_DIM": HIDDEN_DIM,
    "PATIENCE": PATIENCE,
    "ALPHA":ALPHA,
    "LR": LR,
    "normalisation_TOF_RAW": normalisation_TOF_RAW,
    "normalisation_TOF_THM": normalisation_TOF_THM,
    "attention_for_fusion": attention_for_fusion,
    "attention_pooled": attention_pooled,
    "add_tof_features_to_thm": ADD_TOF_TO_THM,
    "C_TOF_RAW": C_TOF_RAW,
    "IMU_FEATURES": selected_features,
    "THM-TOF FEATURES": selected_tof + [f for f in cols['thm'] if 'thm_5' not in f],
    "TOF-RAW FEATURES": raw_tof_sorted,
    "loss_GAMMA": GAMMA,
    "loss_LAMBDA": LAMB,
    "additionnal_IMU_loss": L_IMU,
    "N_CLASSES": len(class_weight),
    "imu_dim":len(selected_features),
    "thm_tof_dim":len(selected_tof),
    "tof_raw_dim":len(raw_tof_sorted),
    "scheduler": SCHEDULER if SCHEDULER else None,
    "factor_scheduler": factor_scheduler if SCHEDULER else None,
    "patience_scheduler": patience_scheduler if SCHEDULER else None,
    "p_dropout": p_dropout,
    "p_jitter": p_jitter,
    "p_moda": p_moda,
    "p_rotation": p_rotation,
    "small_rotation": small_rotation, 
    "x_max_angle": x_max_angle,
    "y_max_angle": y_max_angle,
    "axes_rotation": axes_rotation
}

# ------------------------------- DEMO DATA ---------------------------------
 
train_demographics = pd.read_csv(Config.TRAIN_DEMOGRAPHICS_PATH)


# ------------------------------- TRAINING ---------------------------------

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state = SEED_CV_FOLD) #STRATIFIED k-Fold by group (subject_id)

if not ADD_TOF_TO_THM:
    indices_branches['tof'] = []

train_ids = np.array(split_ids['train']['train_sequence_ids']) #seq_id of data sequences 
groups = [split_ids['train']['train_sequence_subject'][seq_id] for seq_id in train_ids] #subject_id of data_sequences
wrong_subjects = ['SUBJ_045235', 'SUBJ_019262']

# idx_spe_seq = np.where(train_ids == 'SEQ_000007')[0]


### LOOP FOR EACH TRAINING FOLD
best_scores = {
    'mixture':[],
    'imu_only':[],
    'imu_tof_thm':[], 
    }
best_scores_inference = []
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    print(f"\n===== FOLD {fold+1}/{N_SPLITS} =====\n")
    reset_seed(SEED)

    # Split data
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    if TRAIN:

        subjects_id = np.array(groups)[train_idx]
        train_seq_ids = train_ids[train_idx]

        ###### Handedness of subjects in train and validation fold
        subjects_fold_train = np.unique(subjects_id)
        subjects_fold_val= np.unique(np.array(groups)[val_idx])
        handedness_train = [train_demographics[train_demographics['subject'] == subject]['handedness'].iloc[0] for subject in subjects_fold_train]
        handedness_val = [train_demographics[train_demographics['subject'] == subject]['handedness'].iloc[0] for subject in subjects_fold_val]
        print(f"number of left-handed (right-handed) subject in train fold {fold + 1} = {np.sum(np.array(handedness_train) == 0)} ({np.sum(np.array(handedness_train) == 1)})")
        print(f"number of left-handed (right-handed) subject in val fold {fold + 1} = {np.sum(np.array(handedness_val) == 0)} ({np.sum(np.array(handedness_val) == 1)})")
        cond_wrong = [wg_sub in subjects_fold_val for wg_sub in wrong_subjects]
        if any(cond_wrong):
            print(f"wrong wrist wearing detected in val fold {fold + 1}: {np.array(wrong_subjects)[cond_wrong]}")

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
                            seqs_by_subject, selected_features, 
                            p_rotation=p_rotation, 
                            small_rotation=small_rotation, 
                            x_rot_range=(0., x_max_angle),
                            y_rot_range=(0., y_max_angle)
                            )
        X_tr, y_tr, count = rotation_augmented(axes = axes_rotation)
        print(f"number of additional rotated features samples: {count}")
        print(f"shape of training data after augmentation (X, y): {X_tr.shape, y_tr.shape}\n")

        # augmenter = Augment()

        # augmenter = Augment(
        #     p_jitter=0.98, sigma=0.033, scale_range=(0.75,1.16),
        #     p_dropout=0.42,
        #     p_moda=0.39, drift_std=0.004, drift_max=0.39    
        # )
        augmenter = Augment(
            p_jitter=p_jitter, sigma=0.033, scale_range=(0.75,1.16),
            p_dropout=p_dropout,
            p_moda=p_moda, drift_std=0.004, drift_max=0.39    
        )
        #########################################

        train_ds = SensorDataset(X_tr, y_tr, imu_dim = len(idx_imu), alpha=ALPHA, augment=augmenter)  ### TRAINING ROTATION AUGMENTED DATA WITH MixUp \alpha 


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
    
        val_ds = SensorDataset(X_val, y_val, imu_dim = len(idx_imu), training=False) ### VALIDATION DATA (NO AUG, NO MixUp)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


    if TRAIN:
        criterion = SoftCrossEntropy(bfrb_classes=bfrb_classes, gamma = GAMMA, lamb = LAMB) # LOSS FUNCTION bfrb_classes=bfrb_classes, gamma = .5, lamb = .5 

        #model = MiniGestureClassifier(imu_dim=X_tr.shape[2], hidden_dim=128, num_classes=len(class_weight)) # MODEL
        model = GlobalGestureClassifier(imu_dim=len(indices_branches['imu']), 
                                        thm_tof_dim=len(indices_branches['thm']) + len(indices_branches['tof']), 
                                        tof_raw_dim=len(indices_branches['tof_raw']),  
                                        hidden_dim=HIDDEN_DIM, 
                                        num_classes=len(class_weight), 
                                        C_TOF_RAW=C_TOF_RAW,
                                        norm_TOF_RAW=normalisation_TOF_RAW,
                                        norm_TOF_THM=normalisation_TOF_THM,
                                        attention_for_fusion=attention_for_fusion,
                                        attention_pooled= attention_pooled
                                        ) # MODEL
        

        #### SAVE HYPERPARAMETERS AND MODEL ARCHI ####
        if fold == 0:
            n_trial = N_TRIAL
            current_date = datetime.now()
            formatted_date = current_date.strftime("%d-%m-%Y")
            formatted_time = datetime.now().strftime("%H%M%S")

            folder_date = 'trials_' + formatted_date
            folder_trial = 'trial_' + str(n_trial)

            EXPORT_DIR = os.path.join(Config.EXPORT_PARAMS_DIR, folder_date, folder_trial)
            file_txt_path = os.path.join(EXPORT_DIR, 'trial_summary.txt')
            file_pkl_path = os.path.join(EXPORT_DIR, 'all_parameters.pkl')
            os.makedirs(EXPORT_DIR, exist_ok=True)  
            save_hparams_and_architecture(all_parameters, model, file_txt_path)
            save_hparams_and_architecture_to_pkl(all_parameters, model, file_pkl_path)
        ##################################################


        optimizer = optim.Adam(model.parameters(), lr=LR) # OPTIMIZER  weight_decay=WD
        if SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor_scheduler, patience=patience_scheduler, verbose=True)
        else:
            scheduler = None

        best_score, best_score_imu_only, best_score_imu_tof_thm = train_model(model, train_loader, val_loader, 
                                 optimizer, criterion, 
                                 EPOCHS,
                                 DEVICE, 
                                 patience=PATIENCE, 
                                 fold = fold, 
                                 split_indices = indices_branches,
                                 scheduler=scheduler,
                                 L_IMU= L_IMU,
                                 seed_CV_fold = SEED_CV_FOLD
                                 )
        best_scores['mixture'].append(best_score)
        best_scores['imu_only'].append(best_score_imu_only)
        best_scores['imu_tof_thm'].append(best_score_imu_tof_thm)
    else:
        print("---- INFERENCE MODE ----")
        processing_dir = Config.EXPORT_DIR
        models_dir = Config.EXPORT_MODELS_PATH
        predictor = EnsemblePredictor(processing_dir, models_dir, DEVICE, all_parameters)
        inverse_map_classes = predictor.inverse_map_classes
        #map_classes = predictor.map_classes
        
        preds_str = predictor.predict(X_val.to(DEVICE), by_fold = fold)
        preds_int = [inverse_map_classes[pred_str] for pred_str in preds_str]
        best_score, _, _ = competition_metric(y_val, preds_int)
    
        best_scores_inference.append(best_score)

print(np.mean(best_scores['mixture']))

if TRAIN:
    log_best_scores(best_scores, file_txt_path)



# for fold, score in enumerate(best_scores):
#     print(f" - Best score for fold {fold}: {score}")

# print(f"mean score for all folds: {np.mean(best_scores)}")

