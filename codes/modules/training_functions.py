from modules.class_models import *
from modules.functions import competition_metric, Config
from sklearn.metrics import recall_score
import os


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, class_weight, bfrb_classes, patience = 50, fold = None):
    model.to(device)
    early_stopper = EarlyStopping(patience=patience, mode='max', restore_best_weights=True, verbose=True)

    best_score = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for inputs, targets in train_loader:
            # print(targets[:5])
            # print(inputs[0, :10, 0])
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) #, phase_adj = inputs[:, :,  -1]
            targets = targets * (1 - 0.2) + (0.2 / len(class_weight))
            loss = criterion(outputs, targets, class_weight, bfrb_classes)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_targets.extend(targets.argmax(1).cpu().numpy())
        

        train_acc, train_binary_recall, train_macro_f1  = competition_metric(train_targets, train_preds)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        bin_preds = []
        bin_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs) #, phase_adj = inputs[:, :,  -1]
                loss = criterion(outputs, targets, class_weight, bfrb_classes)

                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_targets.extend(targets.argmax(1).cpu().numpy())

                mask_bfrb_classes = np.array([idx in bfrb_classes.numpy() for idx in range(len(class_weight))])
                outputs = torch.nn.functional.softmax(outputs, dim=1)
        
                bin_pred = outputs[:, mask_bfrb_classes].sum(1) > 0.4 #torch.stack([, outputs[:, ~mask_bfrb_classes].sum(1)], dim=1) 
                bin_preds.extend(bin_pred.cpu().numpy())

                bin_target = targets[:, mask_bfrb_classes].sum(1) #, targets[:, ~mask_bfrb_classes].sum(1)], dim=1) 
                bin_targets.extend(bin_target.cpu().numpy())
                
        
        val_acc, val_binary_recall, val_macro_f1 = competition_metric(val_targets, val_preds)     #accuracy_score(val_targets, val_preds)
        val_binary_recall = recall_score(bin_targets, bin_preds)
        early_stopper(val_acc, model)
        if early_stopper.best_score > best_score:
            best_score = early_stopper.best_score
            name = "best_model"
            if fold is not None:
                name += f"_fold_{fold}.pth"
            else:
                name += ".pth"
            torch.save(early_stopper.best_model_state, os.path.join(Config.EXPORT_MODELS_PATH, name ))

        if early_stopper.early_stop:
            print("Training stopped early.")
            break
        

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Bin. : {train_binary_recall:.4f}, Macro: {train_macro_f1:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Bin. : {val_binary_recall:.4f}, Macro: {val_macro_f1:.4f}")



    return best_score


def soft_cross_entropy(pred_logits, soft_targets, class_weight, bfrb_classes, gamma = 1.5):
    mask_bfrb_classes = np.array([idx in bfrb_classes.numpy() for idx in range(len(class_weight))])
    outputs = torch.nn.functional.softmax(pred_logits, dim=1)
    bin_pred = torch.stack([ outputs[:, mask_bfrb_classes].sum(1), outputs[:, ~mask_bfrb_classes].sum(1)], dim=1)
    bin_target = torch.stack([soft_targets[:, mask_bfrb_classes].sum(1), soft_targets[:, ~mask_bfrb_classes].sum(1)], dim=1) 
    
    binary_loss = F.kl_div(
    torch.log(bin_pred + 1e-8),  # log-probabilities
    bin_target,
    reduction='batchmean'
    )

    log_probs = F.log_softmax(pred_logits, dim=1)
    #idx_bfrb_classes = bfrb_classes.numpy()
    #soft_targets = soft_targets * class_weight.to(DEVICE).unsqueeze(0)
    #soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)  # re-normalize
    weighted_kl = F.kl_div(log_probs, soft_targets, reduction='batchmean')
    #weighted_kl = weighted_kl * class_weight.to(DEVICE).unsqueeze(0)
    #weighted_kl[:, idx_bfrb_classes] *=  boost_factor
    return weighted_kl + gamma * binary_loss#.sum(dim = 1).mean() #