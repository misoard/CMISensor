import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_channels, hidden_dim = 16, latent_dim=32, drop = 0.3):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 4 * hidden_dim, kernel_size=5, stride=4, padding=2),  # -> (B, 32, L/2)
            nn.ReLU(),
            nn.Dropout(p = drop),
            nn.Conv1d(4 * hidden_dim, 2 * hidden_dim, kernel_size=5, stride=4, padding=2),           # -> (B, 64, L/4)
            nn.ReLU(),
            nn.Dropout(p = drop),
            nn.Conv1d(2* hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),          # -> (B, 128, L/8)
            nn.ReLU(),
            nn.Dropout(p = drop),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),              #nn.AdaptiveAvgPool1d(1),                                         # -> (B, 128, 1)
        )
        self.latent = nn.Linear(28 * hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 28 * hidden_dim) # (B, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),  #(B, 64, 11)
            nn.ReLU(),
            nn.Dropout(p = drop),
            nn.ConvTranspose1d(hidden_dim, 2 * hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),   # (B, 64, 41)
            nn.ReLU(),
            nn.Dropout(p = drop),
            nn.ConvTranspose1d(2 * hidden_dim, 4 * hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Dropout(p = drop),
            nn.ConvTranspose1d(4 * hidden_dim, input_channels, kernel_size=5, stride=4, padding=2, output_padding=1),
            #nn.Tanh()  # Assuming normalized input
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        z = self.encoder(x)
        z = z.reshape(z.shape[0], -1)
        z = self.latent(z)
        x_recon = self.decoder_fc(z)
        x_recon = x_recon.unsqueeze(-1).reshape(-1, self.hidden_dim, 28)
        #print(x_recon.shape)
        x_recon = self.decoder(x_recon)
        return x_recon


class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p = 0.3),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, H]

        attn_scores = self.attn(lstm_out).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T]

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, H]

        return context



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

class OptionalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x, mask):
        # x: [B, T, input_dim] → [B, input_dim, T]
        x = x.permute(0, 2, 1)
        out = self.net(x)  # [B, hidden_dim, T/4]

        # Adjust mask accordingly by downsampling (average pooling)
        # mask: [B, T]
        mask = mask.unsqueeze(1).float()  # [B, 1, T]
        mask = F.avg_pool1d(mask, kernel_size=2, stride=2)  # [B, 1, T/2]
        mask = F.avg_pool1d(mask, kernel_size=2, stride=2)  # [B, 1, T/4]
        mask = mask.squeeze(1)  # [B, T/4]

        out = out * mask.unsqueeze(1)  # [B, hidden_dim, T/4]

        # Normalize by sum of mask per timestep (avoid div zero)
        norm = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
        out = out / norm.unsqueeze(1)  # broadcast on hidden_dim

        return out

class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    def forward(self, x, seq_len):
        # x: [B, n_feats]
        emb = self.net(x)  # [B, hidden_dim]
        # Expand along time dimension to [B, hidden_dim, seq_len]
        emb = emb.unsqueeze(2).expand(-1, -1, seq_len)
        return emb

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * num_modalities, num_modalities)

    def forward(self, features_list):
        # features_list: list of [B, hidden_dim, T]
        concat = torch.cat(features_list, dim=1)  # [B, hidden_dim * M, T]
        concat_t = concat.permute(0, 2, 1)        # [B, T, hidden_dim * M]
        gate_weights = torch.sigmoid(self.gate(concat_t))  # [B, T, M]

        gated_feats = []
        for i, f in enumerate(features_list):
            f_t = f.permute(0, 2, 1)  # [B, T, hidden_dim]
            w = gate_weights[:, :, i].unsqueeze(-1)  # [B, T, 1]
            gated_feats.append(f_t * w)
        fused = sum(gated_feats)  # [B, T, hidden_dim]
        return fused.permute(0, 2, 1)  # [B, hidden_dim, T]

class GestureClassifier(nn.Module):
    def __init__(self, imu_dim, hidden_dim, num_classes, tof_dim = None, thm_dim = None): # tabular_dim = None
        super().__init__()

        if tof_dim is None:
            tof_dim = imu_dim
        if thm_dim is None:
            thm_dim = imu_dim
        # if tabular_dim is None:
        #     tabular_dim = imu_dim

        self.imu_encoder = IMUEncoder(imu_dim, hidden_dim)
        self.tof_encoder = OptionalEncoder(tof_dim, hidden_dim)
        self.thm_encoder = OptionalEncoder(thm_dim, hidden_dim)
        #self.tabular_encoder = TabularEncoder(tabular_dim, hidden_dim)
        self.fusion = GatedFusion(hidden_dim, num_modalities=3)

        self.classifier_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim, num_classes),
            #nn.Softmax()
        )

    def forward(self, imu, tof=None, thm=None): #, tabular_feats=None
        B, T, _ = imu.shape

        imu_feat = self.imu_encoder(imu)  # [B, hidden_dim, T/4]

        if tof is None:
            tof = torch.zeros_like(imu)
            tof_mask = torch.zeros(B, T, device=imu.device)
        else:
            tof_mask = (~torch.isnan(tof).any(dim=2)).float()

        if thm is None:
            thm = torch.zeros_like(imu)
            thm_mask = torch.zeros(B, T, device=imu.device)
        else:
            thm_mask = (~torch.isnan(thm).any(dim=2)).float()

        tof_feat = self.tof_encoder(tof, tof_mask)  # [B, hidden_dim, T/4]
        thm_feat = self.thm_encoder(thm, thm_mask)  # [B, hidden_dim, T/4]

        # if tabular_feats is None:
        #     tabular_feats = torch.zeros(B, self.tabular_encoder.net[0].in_features, device=imu.device)

        # tab_feat = self.tabular_encoder(tabular_feats, imu_feat.shape[2])  # expand to [B, hidden_dim, T/4]

        fused =  self.fusion([imu_feat, tof_feat, thm_feat])  # [B, hidden_dim, T/4]

        fused_t = fused.permute(0, 2, 1)  # [B, T/4, hidden_dim]

        rnn_out, _ = self.classifier_rnn(fused_t)  # [B, T/4, hidden_dim*2]

        pooled = rnn_out.mean(dim=1)#   # [B, hidden_dim*2]
        pooled = F.dropout(pooled, p=0.5, training=self.training) 

        out = self.classifier_head(pooled)  # [B, num_classes]

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
    
class MiniGestureLSTMClassifier(nn.Module):
    def __init__(self, imu_dim, imu_dim_lstm, hidden_dim, lstm_hidden_dim, num_classes):
        super().__init__()

        self.imu_encoder = IMUEncoder(imu_dim, hidden_dim)
        self.lstm_attn = LSTMWithAttention(imu_dim_lstm, lstm_hidden_dim)
        
        fused_dim = hidden_dim + lstm_hidden_dim

        self.classifier_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim, num_classes),
            #nn.Softmax()
        )

    def forward(self, imu): #, phase_adj = None,
        #B, T, _ = imu.shape

        imu_cnn_out = self.imu_encoder(imu)  # [B, hidden_dim, T]
        imu_pooled = imu_cnn_out.mean(dim=2) # [B, hidden_dim]
        imu_lstm_out = self.lstm_attn(imu)  # [B, H]

        fused = torch.cat([imu_pooled, imu_lstm_out], dim=1)  # [B, hidden_dim + H]
        out = self.classifier_head(fused)

        return out
    

class EarlyStopping:
    def __init__(self, patience=5, mode='max', restore_best_weights=True, verbose=False):
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score, model):
        if self.mode == 'max':
            score_improved = self.best_score is None or current_score > self.best_score
        else:  # 'min'
            score_improved = self.best_score is None or current_score < self.best_score

        if score_improved:
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"EarlyStopping: Improvement found, saving model with score {current_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping: Stopping early.")
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)



class SensorDataset(Dataset):
    def __init__(self, X, y, imu_dim, alpha = None, augment = None, training = True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.alpha = alpha
        self.augment = augment
        self.training = training
        self.imu_dim = imu_dim

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, idx):
        x1, y1 = self.X[idx], self.y[idx]
        y1_onehot = torch.nn.functional.one_hot(y1, num_classes=18).float()

        if self.training and self.augment:
            x1 = x1.numpy().copy()
            x1 = self.augment(x1, imu_dim = self.imu_dim)
            x1 = torch.tensor(x1,  dtype=torch.float32)
            
        if self.alpha is not None:
            rand_idx = np.random.randint(0, len(self.X) - 1)
            x2, y2 = self.X[rand_idx], self.y[rand_idx]

            if self.training and self.augment:
                x2 = x2.numpy().copy()
                x2 = self.augment(x2, imu_dim=self.imu_dim)
                x2 = torch.tensor(x2, dtype=torch.float32)

            y2_onehot = torch.nn.functional.one_hot(y2, num_classes=18).float()

            # Generate lambda from Beta distribution and ensure alpha > 0.
            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(0, min(1, lam))

            x1 = lam * x1 + (1 - lam) * x2
            y1_onehot = lam * y1_onehot + (1 - lam) * y2_onehot
        
        return x1, y1_onehot

class TrackingSampler(torch.utils.data.Sampler):
    def __init__(self, base_sampler):
        self.base_sampler = base_sampler
        self.sampled_indices = []

    def __iter__(self):
        self.sampled_indices = list(self.base_sampler)  # Store for external access
        return iter(self.sampled_indices)

    def __len__(self):
        return len(self.base_sampler)

class DeviceRotationAugment:
    def __init__(self,
                X, y, seqs,       
                seqs_by_subject,
                selected_features,
                x_rot_range = (0, 30), # (0, 45)
                y_rot_range = (0, 30), # (0, 45)
                p_rotation = 0.4,
                small_rotation = 2
                ):     
        
        self.features_to_rotate = [
        ['acc_x', 'acc_y', 'acc_z'],
        ['acc_x_world', 'acc_y_world', 'acc_z_world'],
        ['linear_acc_x', 'linear_acc_y', 'linear_acc_z'],
        ['rotvec_x', 'rotvec_y', 'rotvec_z'],
        ['ang_vel_x', 'ang_vel_y', 'ang_vel_z'],
        ['X_world_x', 'X_world_y', 'X_world_z'], 
        ['Y_world_x', 'Y_world_y', 'Y_world_z'],
        ['Z_world_x', 'Z_world_y', 'Z_world_z'],
        ['rot_x', 'rot_y', 'rot_z', 'rot_w']
        ]
        
        self.seqs_by_subject = seqs_by_subject 
        self.p_rotation = p_rotation
        self.selected_features = selected_features
        self.x_rot_range = x_rot_range
        self.y_rot_range = y_rot_range
        self.small_rotation = small_rotation

        self.X =  torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seqs = seqs
        self.count = 0
        self.iter = 2

    def random_angles_by_seq(self):
        unique_subjects = list(self.seqs_by_subject.keys())
        # Assign a consistent random Y angle per subject
        subject_to_angle = {
            subj:  np.random.uniform(*self.x_rot_range) #, np.random.uniform(*self.y_rot_range)) #np.random.choice(y_range)
            for subj in unique_subjects
        }

        random_small_angles_by_subject = {
            subj: np.random.uniform(-self.small_rotation, self.small_rotation, size=len(seqs))
            for subj, seqs in self.seqs_by_subject.items()
        }


        subject_for_seq = {
        seq_id: (i, subj) for subj, seq_ids in self.seqs_by_subject.items() for i, seq_id in enumerate(seq_ids)
        }

        seq_to_angle = {
            seq_id: subject_to_angle[subj] + random_small_angles_by_subject[subj][i] #, subject_to_angle[subj][1] + random_small_angles_by_subject[subj][i])
            for seq_id, (i, subj) in subject_for_seq.items()
        }

        return seq_to_angle

    def apply_rotation(self, 
                       x: torch.tensor, 
                       ax: str, 
                       seq_id: str,
                       seqs_to_angle) -> np.ndarray:
        x_copy = x.numpy().copy()
        rot_x = seqs_to_angle.get(seq_id, 0.0)
        rot_y = rot_x
        if ax == 'x':
            rot = R.from_euler(ax, rot_x, degrees=True)
        if ax == 'y':
            rot = R.from_euler(ax, rot_y, degrees=True)
        if ax == 'z':
            rot = R.from_euler(ax, 180, degrees=True)
        if ax == 'zx':
            rot = R.from_euler('z', 180, degrees=True) *  R.from_euler('x', rot_x, degrees=True)
        if ax == 'zy':
            rot = R.from_euler('z', 180, degrees=True) *  R.from_euler('y', rot_y, degrees=True)
        if ax == 'xy':
            rot = R.from_euler('x', rot_x, degrees=True) *  R.from_euler('y', rot_y, degrees=True)
        if ax == 'zxy':
            rot = R.from_euler('z', 180, degrees=True) * R.from_euler('x', rot_x, degrees=True) *  R.from_euler('y', rot_y, degrees=True)

        for feats in self.features_to_rotate:
            idx_rotate = np.where(np.isin(self.selected_features, feats))[0]
            if len(idx_rotate) == 0:
                continue

            if not any('rot_' in f for f in feats):
                rotated = rot.apply(x_copy[:, idx_rotate])
                x_copy[:, idx_rotate] = rotated
            else:
                init_quat = x_copy[:, idx_rotate]
                mask = np.linalg.norm(init_quat, axis=1) < 1e-6
                R_orig = R.from_quat(init_quat[~mask])
                R_new = rot * R_orig
                new_quat = np.zeros_like(init_quat)
                new_quat[~mask] = R_new.as_quat()
                x_copy[:, idx_rotate] = new_quat

        return x_copy
    
    # ---------- master call ----------
    def __call__(self,
                 axes: list) -> np.ndarray:
    

        seqs_to_angle = self.random_angles_by_seq()
        
        augmented_X_tr = []
        augmented_y_tr = []

        for xx, yy, seq_id in zip(self.X, self.y, self.seqs):
            augmented_X_tr.append(xx)
            augmented_y_tr.append(yy)

            # Reverse time (assuming time is dimension 0)
            x_rotated = []
            axes_choice = np.array(axes)
            for i in range(self.iter): #self.iter
                #if (np.random.random() < self.p_rotation) and (len(axes_choice) > 0):
                ax = axes_choice[i] #np.random.choice(axes_choice)
                x_rotated.append(self.apply_rotation(xx, ax, seq_id, seqs_to_angle)) # subject_id, subject_to_angle)
                    #axes_choice = np.delete(axes_choice, np.where(axes_choice == ax)) 
            if len(x_rotated) > 0:
                self.count += len(x_rotated)
                x_rotated = [torch.tensor(x) for x in x_rotated]
                augmented_X_tr.extend(x_rotated)
                augmented_y_tr.extend([yy] * len(x_rotated))


        #augmented_X_tr = torch.tensor(augmented_X_tr)
        augmented_X_tr = torch.stack(augmented_X_tr)
        augmented_y_tr = torch.tensor(augmented_y_tr)  # Or use torch.stack if already tensors

        # X_aug = my_aug.augment(self.X.numpy())  # shape preserved
        # y_aug = self.y.clone()          # labels remain the same
        
        # X_aug = torch.cat([self.X, torch.from_numpy(X_aug)], dim=0)
        # y_aug = torch.cat([self.y, y_aug], dim=0)
        return augmented_X_tr, augmented_y_tr, self.count


class Augment:
    def __init__(self,
                 p_jitter=0.8, sigma=0.02, scale_range=[0.9,1.1],
                 p_dropout=0.3,
                 p_moda=0.5,
                 drift_std=0.005,     
                 drift_max=0.25):      
        self.p_jitter  = p_jitter
        self.sigma     = sigma
        self.scale_min, self.scale_max = scale_range
        self.p_dropout = p_dropout
        self.p_moda    = p_moda

        self.drift_std = drift_std
        self.drift_max = drift_max


    # ---------- Jitter & Scaling ----------
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min,
                                   self.scale_max,
                                   size=(1, x.shape[1]))
        return (x + noise) * scale

    # ---------- Sensor Drop-out ----------
    def sensor_dropout(self,
                       x: np.ndarray,
                       imu_dim: int) -> np.ndarray:

        if np.random.random() < self.p_dropout:
            x[:, imu_dim:] = 0.0
        return x

    def motion_drift(self, x: np.ndarray, imu_dim: int) -> np.ndarray:

        T = x.shape[0]

        drift = np.cumsum(
            np.random.normal(scale=self.drift_std, size=(T, 1)),
            axis=0
        )
        drift = np.clip(drift, -self.drift_max, self.drift_max)   

        x[:, :6] += drift

        if imu_dim > 6:
            x[:, 6:imu_dim] += drift     
        return x
    

    
    # ---------- master call ----------
    def __call__(self,
                 x: np.ndarray,
                 imu_dim: int) -> np.ndarray:
        
        if np.random.random() < self.p_jitter:
            x = self.jitter_scale(x)

        if np.random.random() < self.p_moda:
            x = self.motion_drift(x, imu_dim)

        x = self.sensor_dropout(x, imu_dim)
        return x