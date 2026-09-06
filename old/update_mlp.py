import re
with open('/media/storage_data/ai2lab/choemj/STORM/train_probing_mlp.py', 'r') as f:
    content = f.read()

# 1. Imports
content = content.replace('from collections import Counter', 'from collections import Counter\nfrom scipy.optimize import linear_sum_assignment')

# 2. Add labeling logic
label_logic = """
def _is_submarine(lane_idx, ram_state):
    return ram_state[89 + lane_idx] % 8 > 3 and ram_state[89 + lane_idx] % 8 < 7

def extract_diver_set_labels(ram):
    N = len(ram)
    labels = np.zeros((N, 4, 3), dtype=np.float32)
    for idx in range(N):
        current_ram = ram[idx]
        diver_count = 0
        for i in range(4):
            lane_idx = 3 - i
            diver_x = current_ram[74 - i]
            if 0 < diver_x < 160:
                if not _is_submarine(lane_idx, current_ram):
                    diver_y = 141 - lane_idx * 24
                    labels[idx, diver_count, 0] = 1.0  # presence
                    labels[idx, diver_count, 1] = float(diver_x) / 160.0  # Normalize X
                    labels[idx, diver_count, 2] = float(diver_y) / 210.0  # Normalize Y
                    diver_count += 1
    return labels

PROBING_TASKS = {
    "diver_count": {
        "description": "하단 잠수부 개수 분류 (0~6)",
        "type": "classification",
        "num_classes": 7,
        "extract_label": lambda ram: ram[:, 62].astype(np.int64),  # RAM[62] = diver count
    },
    "diver_set": {
        "description": "다중 잠수부 집합 예측 (Bipartite Matching)",
        "type": "set_prediction",
        "num_classes": 12,
        "extract_label": extract_diver_set_labels,
    }
}
"""
content = re.sub(r'PROBING_TASKS = \{.*?\n\}', label_logic.strip(), content, flags=re.DOTALL)

# 3. Add Bipartite loss
loss_logic = """
def bipartite_matching_loss(preds, targets):
    B = preds.shape[0]
    preds = preds.view(B, 4, 3)
    
    total_bce = 0.0
    total_mse = 0.0
    num_assigned = 0
    
    for b in range(B):
        pred_b = preds[b]
        tgt_b = targets[b]
        
        valid_mask = tgt_b[:, 0] == 1.0
        valid_tgts = tgt_b[valid_mask]
        K = valid_tgts.shape[0]
        
        pred_pres_logits = pred_b[:, 0]
        pred_pres_probs = torch.sigmoid(pred_pres_logits)
        pred_coords = pred_b[:, 1:3]
        
        if K == 0:
            bce = F.binary_cross_entropy_with_logits(pred_pres_logits, torch.zeros_like(pred_pres_logits))
            total_bce += bce
            continue
            
        cost_pres = -pred_pres_probs.unsqueeze(1).expand(4, K)
        tgt_coords = valid_tgts[:, 1:3]
        cost_coords = torch.cdist(pred_coords, tgt_coords, p=1)
        
        C = cost_pres + 5.0 * cost_coords
        C = C.detach().cpu().numpy()
        
        row_ind, col_ind = linear_sum_assignment(C)
        
        pres_target = torch.zeros(4, device=preds.device)
        pres_target[row_ind] = 1.0
        
        bce = F.binary_cross_entropy_with_logits(pred_pres_logits, pres_target)
        total_bce += bce
        
        matched_pred_coords = pred_coords[row_ind]
        matched_tgt_coords = tgt_coords[col_ind]
        mse = F.mse_loss(matched_pred_coords, matched_tgt_coords, reduction='sum')
        total_mse += mse
        num_assigned += K
        
    avg_bce = total_bce / B
    avg_mse = total_mse / max(num_assigned, 1)
    loss = avg_bce + 5.0 * avg_mse
    return loss, avg_bce, avg_mse

def train_probe(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cuda", task_type="classification"):
"""
content = content.replace('def train_probe(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cuda"):', loss_logic.strip())

# 4. Modify train loop inside train_probe
train_loop = """
        model.train()
        train_loss = 0.0
        train_metric = 0.0
        train_total = 0
        for z, labels in train_loader:
            z, labels = z.to(device), labels.to(device)
            logits = model(z)
            
            if task_type == "classification":
                loss = criterion(logits, labels)
                train_metric += (logits.argmax(dim=-1) == labels).sum().item()
            else:
                loss, bce, mse = bipartite_matching_loss(logits, labels)
                train_metric += mse.item() * len(labels) # store MSE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_total += len(labels)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        val_total = 0
        with torch.no_grad():
            for z, labels in val_loader:
                z, labels = z.to(device), labels.to(device)
                logits = model(z)
                
                if task_type == "classification":
                    loss = criterion(logits, labels)
                    val_metric += (logits.argmax(dim=-1) == labels).sum().item()
                else:
                    loss, bce, mse = bipartite_matching_loss(logits, labels)
                    val_metric += mse.item() * len(labels)

                val_loss += loss.item() * len(labels)
                val_total += len(labels)

        if task_type == "classification":
            val_acc = 100.0 * val_metric / val_total
            train_acc = 100.0 * train_metric / train_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            metric_str = f"Train Acc: {train_acc:.1f}% | Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.1f}%"
        else:
            val_mse = val_metric / val_total
            train_mse = train_metric / train_total
            # For regression, lower MSE is better. Let's negate it so best_val_acc logic works, or redefine it.
            if best_val_acc == 0.0 or val_mse < best_val_acc:
                best_val_acc = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            metric_str = f"Train MSE: {train_mse:.4f} | Val Loss: {val_loss/val_total:.4f}, Val MSE: {val_mse:.4f}"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss/train_total:.4f}, {metric_str}")
"""
# Replace the train/val loop
start_idx = content.find('for epoch in range(1, num_epochs + 1):')
end_idx = content.find('return best_state, best_val_acc')
content = content[:start_idx] + 'for epoch in range(1, num_epochs + 1):' + train_loop + '    return best_state, best_val_acc' + content[end_idx+len('return best_state, best_val_acc'):]


# 5. Modify evaluation
eval_logic = """
def evaluate_per_class(model, dataset, num_classes, device="cuda", task_type="classification"):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    if task_type == "classification":
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for z, labels in loader:
                z, labels = z.to(device), labels.to(device)
                preds = model(z).argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                for c in range(num_classes):
                    mask = labels == c
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        overall_acc = 100.0 * (all_preds == all_labels).mean()
        return class_correct, class_total, overall_acc
    else:
        total_mse = 0.0
        total_assigned = 0
        total_bce = 0.0
        batches = 0
        with torch.no_grad():
            for z, labels in loader:
                z, labels = z.to(device), labels.to(device)
                logits = model(z)
                loss, bce, mse = bipartite_matching_loss(logits, labels)
                # count objects
                total_mse += mse.item()
                total_bce += bce.item()
                batches += 1
        return None, None, total_mse / batches  # Return Avg MSE
"""
content = re.sub(r'def evaluate_per_class.*?return class_correct, class_total, overall_acc', eval_logic.strip(), content, flags=re.DOTALL)

# 6. Change labels_tensor type creation in main()
content = content.replace('labels_tensor = torch.from_numpy(labels).long()', 'labels_tensor = torch.from_numpy(labels).float() if task_info["type"] == "set_prediction" else torch.from_numpy(labels).long()')

content = content.replace('num_epochs=args.epochs, lr=args.lr, device="cuda"', 'num_epochs=args.epochs, lr=args.lr, device="cuda", task_type=task_info["type"]')
content = content.replace('task_info["num_classes"]', 'task_info["num_classes"], task_type=task_info["type"]')

# 7. Modify report logic
report_logic = """
    if task_info["type"] == "classification":
        print(f"\n{'='*60}")
        print(f"  Overall Accuracy: {colorama.Fore.GREEN}{overall_acc:.1f}%{colorama.Style.RESET_ALL}")
        print(f"  Best Val Accuracy: {colorama.Fore.GREEN}{best_val_acc:.1f}%{colorama.Style.RESET_ALL}")
        print(f"\n  Per-class accuracy:")
        for c in range(task_info["num_classes"]):
            if class_total[c] > 0:
                acc = 100.0 * class_correct[c] / class_total[c]
                bar = "█" * int(acc / 2)
                print(f"    class {c} (n={int(class_total[c]):5d}): {acc:5.1f}%  {bar}")
            else:
                print(f"    class {c} (n=    0):   N/A  (데이터 없음)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"  Overall Avg MSE (normalized): {colorama.Fore.GREEN}{overall_acc:.5f}{colorama.Style.RESET_ALL}")
        print(f"  (Note: X, Y coordinates were normalized to [0, 1])")
        print(f"{'='*60}")
"""
content = re.sub(r'print\(f"\\n\{\'=\'\*60\}"\)\n    print\(f"  Overall Accuracy:.*?print\(f"\{\'=\'\*60\}"\)', report_logic.strip(), content, flags=re.DOTALL)

with open('/media/storage_data/ai2lab/choemj/STORM/train_probing_mlp.py', 'w') as f:
    f.write(content)
