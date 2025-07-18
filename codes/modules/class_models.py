import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from modules.functions import *
import glob
from collections import Counter


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
        # self.attn = nn.Sequential(
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        #     nn.Tanh(),
        #     nn.Conv1d(hidden_dim, 1, kernel_size=1)
        # )
        # self.attn = nn.Sequential(
        #     nn.Conv1d(hidden_dim, 1, kernel_size=1),  # [B, 1, T]
        #     nn.Softmax(dim=-1)
        # )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),  # [B, 1, T]
            nn.Tanh()
        )
        self.bias_strength = bias_strength
        self.weights = None

    def forward(self, x, phase_adj = None):
        # x: [B, hidden_dim, T]
        scores = self.attn(x.permute(0, 2, 1)).squeeze(-1)  # [B, T]

        if phase_adj is not None:
            #bias = (phase_adj.float() * self.bias_strength)  # [B, T]
            scores = scores #+ bias

        weights = F.softmax(scores, dim=1)  # [B, T]
        #weights = scores
        self.weights = weights
        pooled = torch.sum(x * weights.unsqueeze(1), dim=2)  # [B, hidden_dim]
        return pooled

class IMUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), #inplace=True
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.3), 
            #nn.MaxPool1d(kernel_size=2, stride=2),  # halves time length
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            #nn.Dropout(p=0.2), 
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.3),
            #nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(hidden_dim),
            #nn.Dropout(p=0.2),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x, mask):
        # x: [B, T, input_dim] → [B, input_dim, T]
        x = x.permute(0, 2, 1)
        out = self.net(x)  # [B, hidden_dim, T/4]

        # Adjust mask accordingly by downsampling (average pooling)
        # mask: [B, T]
        mask = mask.unsqueeze(1).float()  # [B, 1, T]
        #mask = F.avg_pool1d(mask, kernel_size=2, stride=2)  # [B, 1, T/2]
        #mask = F.avg_pool1d(mask, kernel_size=2, stride=2)  # [B, 1, T/4]
        mask = mask.squeeze(1)  # [B, T/4]

        out = out * mask.unsqueeze(1)  # [B, hidden_dim, T/4]

        #Normalize by sum of mask per timestep (avoid div zero)
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
    

class TOFEncoder3DWithSpatialAttention(nn.Module):
    def __init__(self, in_channels=5, out_channels=64, hidden_dim=128, H=8, W=8):
        super().__init__()

        # 3D CNN to process [B, 5, T, 8, 8]
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial attention: per pixel over each 8x8 grid at each time step
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),  # attention logits per pixel
        )

        self.H = H
        self.W = W

    def forward(self, x):
        """
        x: [B, 5, T, 8, 8]
        returns: [B, hidden_dim, T] (aggregated per timestep)
        """
        B, C, T, H, W = x.shape

        # Apply 3D conv
        feat = self.conv3d(x)  # [B, hidden_dim, T, H, W]

        # Reshape for spatial attention per time step
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        feat_flat = feat_reshaped.view(B * T, -1, H, W)  # [B*T, C, H, W]

        # Spatial attention scores
        attn_logits = self.spatial_attn(feat_flat)  # [B*T, 1, H, W]
        attn_scores = F.softmax(attn_logits.view(B * T, -1), dim=-1).view(B * T, 1, H, W)  # [B*T, 1, H, W]

        # Apply attention
        weighted_feat = feat_flat * attn_scores  # [B*T, C, H, W]
        aggregated = weighted_feat.view(B, T, -1, H * W).sum(dim=-1)  # [B, T, C]

        # Transpose to [B, C, T] to match other branches
        aggregated = aggregated.permute(0, 2, 1)

        return aggregated  # [B, hidden_dim, T]




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

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities=2):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modalities)])
        self.values = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modalities)])
        self.scale = hidden_dim ** 0.5

    def forward(self, inputs):
        """
        inputs: list of [B, hidden_dim, T] tensors
        output: [B, hidden_dim, T] fused tensor
        """
        # Compute shared query
        #stacked_inputs = torch.stack(inputs, dim=1)  # [B, M, hidden_dim, T]
        #B, M, C, T = stacked_inputs.shape

        query = self.query(inputs[0].permute(0, 2, 1))  # [B, T, hidden_dim]
        keys = [key(x.permute(0, 2, 1)) for key, x in zip(self.keys, inputs)]   # each: [B, T, hidden_dim]
        values = [val(x.permute(0, 2, 1)) for val, x in zip(self.values, inputs)]  # each: [B, T, hidden_dim]

        key_tensor = torch.stack(keys, dim=1)     # [B, M, T, hidden_dim]
        value_tensor = torch.stack(values, dim=1) # [B, M, T, hidden_dim]

        # Attention: dot product over modalities
        attn_scores = torch.einsum('btc,bmtc->btm', query, key_tensor) / self.scale  # [B, T, M]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, M]

        # Weighted sum over modalities
        fused = torch.einsum('btm,bmtc->btc', attn_weights, value_tensor)  # [B, T, hidden_dim]
        return fused.permute(0, 2, 1)  # back to [B, hidden_dim, T]



class GestureClassifier(nn.Module):
    def __init__(self, imu_dim, hidden_dim, num_classes, tof_dim = None, thm_dim = None): # tabular_dim = None
        super().__init__()

        if tof_dim is None:
            tof_dim = imu_dim
        if thm_dim is None:
            thm_dim = imu_dim

        self.imu_encoder = IMUEncoder(imu_dim, hidden_dim)
        self.tof_encoder = OptionalEncoder(tof_dim, hidden_dim)
        self.thm_encoder = OptionalEncoder(thm_dim, hidden_dim)
        self.fusion = GatedFusion(hidden_dim, num_modalities=3)


        self.classifier_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 , hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim, num_classes),
            #nn.Softmax()
        )

    def forward(self, imu, thm=None, tof=None): #, tabular_feats=None
        B, T, _ = imu.shape

        imu_feat = self.imu_encoder(imu)  # [B, hidden_dim, T/4]

        # nan_mask = torch.isnan(thm)
        # nan_indices = torch.nonzero(nan_mask, as_tuple=True)[0].detach().cpu()
        # print(f"number of NaN (detect possible FE errors): {len(np.unique(nan_indices.numpy()))}")

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

        fused =  self.fusion([imu_feat, tof_feat, thm_feat])  # [B, hidden_dim, T/4]

        fused_t = fused.permute(0, 2, 1)  # [B, T/4, hidden_dim]
        rnn_out, _ = self.classifier_rnn(fused_t)  # [B, T/4, hidden_dim*2]

        pooled = rnn_out.mean(dim=1)#   # [B, hidden_dim*2]
        #pooled = F.dropout(pooled, p=0.5, training=self.training) 

        out = self.classifier_head(pooled)  # [B, num_classes]

        return out
    

class GlobalGestureClassifier(nn.Module):
    def __init__(self, imu_dim, hidden_dim, num_classes, thm_tof_dim = None, p_drop_tof = 0.): # tabular_dim = None
        super().__init__()

        if thm_tof_dim is None:
            thm_tof_dim = imu_dim
        
        self.thm_tof_dim = thm_tof_dim

        self.imu_encoder = IMUEncoder(imu_dim, hidden_dim)
        self.attn_pool = AttentionPooling(hidden_dim)

        self.thm_tof_encoder = OptionalEncoder(thm_tof_dim, hidden_dim)

        self.gated_fusion = GatedFusion(hidden_dim, num_modalities=2)
        self.attention_fusion = AttentionFusion(hidden_dim, num_modalities=2)
        self.final_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        #self.bilstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        #self.classifier_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.p_drop_tof = p_drop_tof

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim, num_classes),
            #nn.Softmax()
        )

        self.imu_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, imu, thm_tof=None): #, tabular_feats=None
        B, T, _ = imu.shape

        imu_feat = self.imu_encoder(imu)  # [B, hidden_dim, T]
        pooled_imu = imu_feat.mean(dim= 2)
        logits_imu = self.imu_head(imu_feat)

        if thm_tof is None:
            thm_tof = torch.zeros_like(imu)
            tof_mask = torch.zeros(B, T, device=imu.device)
        else:
            tof_mask = (~torch.isnan(thm_tof).any(dim=2)).float()

        # #tof_mask_drop = torch.ones_like(tof_mask)
        # if self.training and torch.rand(1).item() < self.p_drop_tof:
        #     thm_tof = torch.zeros_like(thm_tof)
        #     #tof_mask_drop = torch.zeros(B, T, device=imu.device)
    

        thm_tof_feat = self.thm_tof_encoder(thm_tof, tof_mask)  # [B, hidden_dim, T]
        attn =  self.attention_fusion([imu_feat, thm_tof_feat])  # [B, hidden_dim, T]
        gated =  self.gated_fusion([imu_feat, thm_tof_feat])  # [B, hidden_dim, T]
        

        alpha = torch.sigmoid(self.alpha).view(1, -1, 1)  # shape [1, C, 1]
        # if thm_tof is None or tof_mask_drop.sum() == 0:
        #     print("houla")
        #     alpha = torch.ones_like(alpha)  # full weight to gated (IMU)
        
        fused = alpha * gated + (1 - alpha) * attn  # broadcast over B, T
        pooled_fused= self.attn_pool(fused) #fused.mean(dim= 2) #
        #mean_pooled = gated.mean(dim=2)
        #max_pooled = gated.max(dim=2).values

        #pooled_imu = F.dropout(pooled_imu, p=0.2, training=self.training)
        #pooled_fused = F.dropout(pooled_fused, p=0.2, training=self.training)

        pooled = self.final_fusion(torch.cat([pooled_imu, pooled_fused], dim=1))
        #pooled = torch.cat([pooled_imu, pooled_fused], dim=1)
        out = self.classifier_head(pooled)  # [B, num_classes]

        return out, logits_imu   

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
    def __init__(self, patience=5, mode='max', restore_best_weights=True, verbose=False, logger = None):
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.logger = logger

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
                if self.logger is not None:
                    self.logger.info(f"EarlyStopping: Improvement found, saving model with score {current_score:.4f}")
                else:
                    print(f"EarlyStopping: Improvement found, saving model with score {current_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                if self.logger is not None:
                    self.logger.info(f"EarlyStopping: No improvement for {self.counter} epoch(s)")
                else:
                    print(f"EarlyStopping: No improvement for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    if self.logger is not None:
                        self.logger.info("EarlyStopping: Stopping early.")
                    else:
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
                ax = axes_choice[i] #np.random.choice(axes_choice) # #
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
    
    def predict(self, torch_seq, by_fold = None):
    # torch_seq: [N, ...]  (N = batch size)

        if by_fold is None:
            pred_by_model = []
    
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    output = model(torch_seq)  # [N, num_classes]
                    preds = output.argmax(1).cpu().numpy()  # shape: [N]
                    pred_by_model.append(preds)  # list of arrays
        
            # Transpose to get predictions per sample:
            # pred_by_model: list of model predictions → shape: [num_models, N]
            # after zip(*...), we get: [ [model1_pred_sample1, model2_pred_sample1, ...], ... ]
            pred_by_model = list(zip(*pred_by_model))  # shape: [N, num_models]
        
            final_preds = []
            for sample_preds in pred_by_model:
                most_common_prediction = Counter(sample_preds).most_common(1)[0][0]
                final_preds.append(str(self.map_classes[most_common_prediction]))
        else:
            model = self.models[by_fold]
            model.eval()
            with torch.no_grad():
                output = model(torch_seq)  # [N, num_classes]
                preds = output.argmax(1).cpu().numpy()  # shape: [N]
            final_preds = [str(self.map_classes[pred]) for pred in preds]
        
        if len(final_preds) == 1:
            return final_preds[0]
        else:
            return final_preds  # length N list of mapped predictions