from modules.class_models import *
from modules.functions import competition_metric, Config
from modules.functions import DEVICE
from sklearn.metrics import recall_score
import os


def train_model(model, 
                train_loader, val_loader, 
                optimizer, criterion, 
                epochs, batch_size, 
                device, 
                patience = 50, 
                fold = None, 
                logger = None, 
                split_indices = None, 
                scheduler = None, 
                hide_val_half = True,
                L_IMU = 0.2
                ):
    reset_seed(42)
    model.to(device)
    early_stopper = EarlyStopping(patience=patience, mode='max', restore_best_weights=True, verbose=True, logger = logger)
    if split_indices is not None:
        idx_thm_tof = list(split_indices['thm']) + list(split_indices['tof'])
    
    if logger is not None:
        logger.info(f"lengths features: \
                            {len(split_indices['imu'])} (IMU) \
                            {len(idx_thm_tof)} (TOF-THM) \
                            {len(split_indices['tof_raw'])} (TOF-RAW) \
                            ")
    else:
        print(f"lengths features: \
                            {len(split_indices['imu'])} (IMU) \
                            {len(idx_thm_tof)} (TOF-THM) \
                            {len(split_indices['tof_raw'])} (TOF-RAW) \
                            ")
    best_score = 0
    best_score_imu_only = 0
    best_score_imu_tof_thm = 0
    i_scheduler = 0
    for epoch in range(1, epochs + 1):
        #check_memory()
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for inputs, targets in train_loader:

            # if hide_val_half and split_indices is not None:
            #     half = batch_size // 2
            #     x_front = inputs[:half]               
            #     x_back  = inputs[half:].clone()   
            #     x_back[:, :, idx_thm_tof] = 0.0    
            #     inputs = torch.cat([x_front, x_back], dim=0)
            # print(targets[:5])
            # print(inputs[0, :10, 0])
            inputs, targets = inputs.to(device), targets.to(device)
            #check_memory()
            optimizer.zero_grad()
            if split_indices is not None:
                outputs, imu_logits = model(inputs[:, :, split_indices['imu']], inputs[:, :, idx_thm_tof], inputs[:, :, split_indices['tof_raw']]) #, phase_adj = inputs[:, :,  -1]
            else:
                outputs = model(inputs) #, phase_adj = inputs[:, :,  -1]
            #check_memory()
            #targets = targets * (1 - 0.1) + (0.1 / 18)
            imu_loss = criterion(imu_logits, targets)
            loss = criterion(outputs, targets) #, class_weight, bfrb_classes)
            loss += L_IMU * imu_loss
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step(i_scheduler)
                i_scheduler +=1

            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_targets.extend(targets.argmax(1).cpu().numpy())
        

        train_acc, _, train_macro_f1  = competition_metric(train_targets, train_preds)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_preds = {'out': [], 'imu_only': [], 'all': []}
        val_targets = {'out': [], 'imu_only': [], 'all': []}
        # bin_preds = []
        # bin_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                #check_memory()
                if hide_val_half and split_indices is not None:
                    half = min(batch_size // 2, inputs.shape[0] // 2)
                    x_front = inputs[:half]               
                    x_back  = inputs[half:].clone()   
                    x_back[:, :, idx_thm_tof] = 0.0    
                    inputs = torch.cat([x_front, x_back], dim=0)
                    x_back, x_front = x_back.to(device), x_front.to(device)


                inputs, targets = inputs.to(device), targets.to(device)
                if split_indices is not None:
                    outputs, imu_logits = model(inputs[:, :, split_indices['imu']], inputs[:, :, idx_thm_tof], inputs[:, :, split_indices['tof_raw']]) 
                    assert x_back[:, :, split_indices['imu']].shape[2] > 0, "IMU split is empty!"
                    outputs_imu_only, _ = model(x_back[:, :, split_indices['imu']], x_back[:, :, idx_thm_tof], x_back[:, :, split_indices['tof_raw']]) 
                    outputs_all, _ = model(x_front[:, :, split_indices['imu']], x_front[:, :, idx_thm_tof], x_front[:, :, split_indices['tof_raw']]) 
                else:
                    outputs = model(inputs) #, phase_adj = inputs[:, :,  -1]               
                
                loss = criterion(outputs, targets) #, class_weight, bfrb_classes)
                imu_loss = criterion(imu_logits, targets) #, class_weight, bfrb_classes)
                loss += L_IMU * imu_loss
                val_loss += loss.item()

                if split_indices is not None:
                    val_preds['all'].extend(outputs_all.argmax(1).cpu().numpy())
                    val_preds['imu_only'].extend(outputs_imu_only.argmax(1).cpu().numpy())

                    val_targets['all'].extend(targets[:half].argmax(1).cpu().numpy())
                    val_targets['imu_only'].extend(targets[half:].argmax(1).cpu().numpy())

                val_preds['out'].extend(outputs.argmax(1).cpu().numpy())
                val_targets['out'].extend(targets.argmax(1).cpu().numpy())


                # mask_bfrb_classes = np.array([idx in bfrb_classes.numpy() for idx in range(outputs.shape[1])])
                # outputs = torch.nn.functional.softmax(outputs, dim=1)
        
                # bin_pred = outputs[:, mask_bfrb_classes].sum(1) > 0.4 #torch.stack([, outputs[:, ~mask_bfrb_classes].sum(1)], dim=1) 
                # bin_preds.extend(bin_pred.cpu().numpy())

                # bin_target = targets[:, mask_bfrb_classes].sum(1) #, targets[:, ~mask_bfrb_classes].sum(1)], dim=1) 
                # bin_targets.extend(bin_target.cpu().numpy())
                
        val_acc, _, val_macro_f1 = competition_metric(val_targets['out'], val_preds['out'])     #accuracy_score(val_targets, val_preds)
        early_stopper(val_acc, model)

        #val_binary_recall = recall_score(bin_targets, bin_preds)
        if early_stopper.best_score > best_score:
            best_score = early_stopper.best_score
            name = "best_model"
            if fold is not None:
                name += f"_fold_{fold}.pth"
            else:
                name += ".pth"
            #torch.save(early_stopper.best_model_state, os.path.join(Config.EXPORT_MODELS_PATH, name ))

        
        
        if split_indices is not None:
            val_acc_all, _, _ = competition_metric(val_targets['all'], val_preds['all'])    
            val_acc_imu_only, _, _ = competition_metric(val_targets['imu_only'], val_preds['imu_only'])   
            if logger is not None:
                logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro: {train_macro_f1:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f},  Acc (imu only): {val_acc_imu_only:.4f},  Acc (imu+thm+tof): {val_acc_all:.4f}, Macro: {val_macro_f1:.4f}")
            else:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro: {train_macro_f1:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f},  Acc (imu only): {val_acc_imu_only:.4f},   Acc (imu+thm+tof): {val_acc_all:.4f},  Macro: {val_macro_f1:.4f}")
        
            ### BEST IMU-ONLY MODEL ###
            if  val_acc_imu_only > best_score_imu_only:
                best_score_imu_only = val_acc_imu_only
                name = "best_model_imu_only"
                if fold is not None:
                    name += f"_fold_{fold}.pth"
                else:
                    name += ".pth"
                #torch.save(model.state_dict(), os.path.join(Config.EXPORT_MODELS_PATH, name ))
        
            ### BEST IMU-TOF-THM MODEL ###
            if  val_acc_all > best_score_imu_tof_thm:
                best_score_imu_tof_thm = val_acc_all
                name = "best_model_imu_tof_thm"
                if fold is not None:
                    name += f"_fold_{fold}.pth"
                else:
                    name += ".pth"
                #torch.save(model.state_dict(), os.path.join(Config.EXPORT_MODELS_PATH, name ))
        
        else:
            if logger is not None:
                logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro: {train_macro_f1:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro: {val_macro_f1:.4f}")
            else:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro: {train_macro_f1:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro: {val_macro_f1:.4f}")

        if early_stopper.early_stop:
            if logger is not None:
                logger.info("Training stopped early.")
            else:
                print("Training stopped early.")
            break


    return best_score, best_score_imu_only, best_score_imu_tof_thm 


class SoftCrossEntropy:
    def __init__(self,
                 bfrb_classes = None, gamma = None, lamb = None, class_weights = None, device = DEVICE):      
        self.gamma = gamma
        self.lamb = lamb
        self.class_weights = class_weights
        self.bfrb_classes = bfrb_classes
        self.device = device

    def __call__(self,
                 preds: torch.tensor,
                 soft_targets: torch.tensor
                 ):
        
        outputs = torch.nn.functional.softmax(preds, dim=1)
        preds_log = F.log_softmax(preds, dim=1)


        if self.class_weights is not None:
            soft_targets = soft_targets * self.class_weights.to(self.device).unsqueeze(0)
            soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)  # re-normalize

        weighted_kl = F.kl_div(preds_log, soft_targets, reduction='batchmean')

        if self.bfrb_classes is None and (self.gamma is not None or self.lamb is not None):
            raise ValueError("bfrb_classes should not be None when lamb or gamma is specified")

        if self.bfrb_classes is not None and (self.gamma is not None or self.lamb is not None):
            mask_bfrb_classes = np.array([idx in self.bfrb_classes.numpy() for idx in range(preds.shape[1])])
            

            bfrb_pred = torch.cat( [outputs[:, mask_bfrb_classes], outputs[:, ~mask_bfrb_classes].sum(dim=1, keepdim=True)], dim=1)
            bfrb_target = torch.cat( [soft_targets[:, mask_bfrb_classes], soft_targets[:, ~mask_bfrb_classes].sum(dim=1, keepdim=True)], dim=1)

            bin_pred = torch.stack([ outputs[:, mask_bfrb_classes].sum(1), outputs[:, ~mask_bfrb_classes].sum(1)], dim=1)
            bin_target = torch.stack([soft_targets[:, mask_bfrb_classes].sum(1), soft_targets[:, ~mask_bfrb_classes].sum(1)], dim=1) 

            brfb_loss = F.kl_div(
            torch.log(bfrb_pred + 1e-8),  # log-probabilities
            bfrb_target,
            reduction='batchmean'
            )

            binary_loss = F.kl_div(
            torch.log(bin_pred + 1e-8),  # log-probabilities #torch.log(+1e-8)
            bin_target,
            reduction='batchmean'
            )


            if self.gamma is not None and self.lamb is None:
                return  weighted_kl + self.gamma * brfb_loss 
            if self.gamma is None and self.lamb is not None:
                return  weighted_kl + self.lamb * binary_loss 
            if self.gamma is not None and self.lamb is not None:
                return   weighted_kl + self.gamma * brfb_loss + self.lamb * binary_loss    
        else:
            return weighted_kl


# def soft_cross_entropy(pred_logits, soft_targets, class_weight, bfrb_classes, gamma = 0.7):
#     mask_bfrb_classes = np.array([idx in bfrb_classes.numpy() for idx in range(len(class_weight))])
#     outputs = torch.nn.functional.softmax(pred_logits, dim=1)

#     #soft_targets = soft_targets * class_weight.to(DEVICE).unsqueeze(0)
#     #soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)  # re-normalize

#     bfrb_pred = torch.cat( [outputs[:, mask_bfrb_classes], outputs[:, ~mask_bfrb_classes].sum(dim=1, keepdim=True)], dim=1)
#     bfrb_target = torch.cat( [soft_targets[:, mask_bfrb_classes], soft_targets[:, ~mask_bfrb_classes].sum(dim=1, keepdim=True)], dim=1)

#     bin_pred = torch.stack([ outputs[:, mask_bfrb_classes].sum(1), outputs[:, ~mask_bfrb_classes].sum(1)], dim=1)
#     bin_target = torch.stack([soft_targets[:, mask_bfrb_classes].sum(1), soft_targets[:, ~mask_bfrb_classes].sum(1)], dim=1) 
    
#     binary_loss = F.kl_div(
#     torch.log(bin_pred + 1e-8),  # log-probabilities #torch.log(+1e-8)
#     bin_target,
#     reduction='batchmean'
#     )

#     brfb_loss = F.kl_div(
#     torch.log(bfrb_pred + 1e-8),  # log-probabilities
#     bfrb_target,
#     reduction='batchmean'
#     )

#     #idx_bfrb_classes = bfrb_classes.numpy()
#     targets_log = F.log_softmax(pred_logits, dim=1)
#     weighted_kl = F.kl_div(targets_log, soft_targets, reduction='batchmean')
#     #weighted_kl = weighted_kl * class_weight.to(DEVICE).unsqueeze(0)
#     #weighted_kl[:, idx_bfrb_classes] *=  boost_factor
#     return  gamma * weighted_kl + (1. - gamma) * brfb_loss #gamma * weighted_kl + (1. - gamma) * brfb_loss#.sum(dim = 1).mean() #