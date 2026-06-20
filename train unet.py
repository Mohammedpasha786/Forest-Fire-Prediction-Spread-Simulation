Trains FireRiskUNet on the patched feature stack produced by feature_stack.py.

Usage:
    python src/prediction/train_unet.py

Produces:
    models/fire_unet_best.pt
    outputs/maps/training_curves.png
    outputs/maps/confusion_matrix.png

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.unet_model import FireRiskUNet, count_parameters

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class PatchDataset(Dataset):
    """Extracts overlapping patches from the full feature stack + label grid,
    enabling mini-batch training on a single large scene (standard practice
    for geospatial segmentation where one "image" = one region)."""

    def __init__(self, features, labels, patch_size=256, stride=128, indices=None):
        self.features = features  # (C, H, W)
        self.labels = labels      # (H, W)
        self.patch_size = patch_size
        c, h, w = features.shape
        coords = []
        for y in range(0, max(h - patch_size, 0) + 1, stride):
            for x in range(0, max(w - patch_size, 0) + 1, stride):
                coords.append((y, x))
        if h <= patch_size:
            coords = [(0, x) for (_, x) in coords] or [(0, 0)]
        if w <= patch_size:
            coords = [(y, 0) for (y, _) in coords] or [(0, 0)]
        self.coords = coords if indices is None else [coords[i] for i in indices]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        ps = self.patch_size
        feat_patch = self.features[:, y:y + ps, x:x + ps]
        label_patch = self.labels[y:y + ps, x:x + ps]
        return torch.from_numpy(feat_patch.copy()), torch.from_numpy(label_patch.copy()).long()


def split_indices(n, val_split, test_split, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def compute_metrics(preds, targets, num_classes):
    preds = preds.flatten()
    targets = targets.flatten()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    accuracy = np.trace(cm) / cm.sum()
    precision, recall, f1, iou = [], [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        union = tp + fp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        i = tp / union if union > 0 else 0.0
        precision.append(p); recall.append(r); f1.append(f); iou.append(i)

    total = cm.sum()
    row_marg = cm.sum(axis=1) / total
    col_marg = cm.sum(axis=0) / total
    expected_acc = np.sum(row_marg * col_marg)
    kappa = (accuracy - expected_acc) / (1 - expected_acc) if expected_acc < 1 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision_per_class": [float(v) for v in precision],
        "recall_per_class": [float(v) for v in recall],
        "f1_per_class": [float(v) for v in f1],
        "iou_per_class": [float(v) for v in iou],
        "mean_iou": float(np.mean(iou)),
        "kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
    }


def train(config):
    pred_cfg = config["prediction_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed_dir = os.path.join(ROOT, config["paths"]["processed_dir"])
    features = np.load(os.path.join(processed_dir, "feature_stack.npy"))
    labels = np.load(os.path.join(processed_dir, "label.npy"))
    print(f"Feature stack: {features.shape}, Label grid: {labels.shape}")

    full_ds = PatchDataset(features, labels, pred_cfg["patch_size"], pred_cfg["patch_stride"])
    n_patches = len(full_ds)
    train_idx, val_idx, test_idx = split_indices(
        n_patches, pred_cfg["val_split"], pred_cfg["test_split"],
        config["synthetic_data"]["random_seed"],
    )
    print(f"Patches -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    train_ds = PatchDataset(features, labels, pred_cfg["patch_size"], pred_cfg["patch_stride"], train_idx)
    val_ds = PatchDataset(features, labels, pred_cfg["patch_size"], pred_cfg["patch_stride"], val_idx)
    test_ds = PatchDataset(features, labels, pred_cfg["patch_size"], pred_cfg["patch_stride"], test_idx)

    train_loader = DataLoader(train_ds, batch_size=pred_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=pred_cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=pred_cfg["batch_size"])

    model = FireRiskUNet(
        in_channels=pred_cfg["input_channels"],
        num_classes=pred_cfg["num_classes"],
        base_filters=pred_cfg["base_filters"],
        depth=pred_cfg["depth"],
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Class imbalance is expected (most pixels are nil/low risk) -> weighted loss
    class_counts = np.bincount(labels.flatten(), minlength=pred_cfg["num_classes"]).astype(np.float64)
    class_weights = (class_counts.sum() / (class_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * pred_cfg["num_classes"]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=pred_cfg["learning_rate"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    models_dir = os.path.join(ROOT, config["paths"]["models_dir"])
    os.makedirs(models_dir, exist_ok=True)
    best_path = os.path.join(models_dir, "fire_unet_best.pt")

    for epoch in range(pred_cfg["epochs"]):
        model.train()
        train_losses = []
        for feat, label in train_loader:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(feat)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for feat, label in val_loader:
                feat, label = feat.to(device), label.to(device)
                logits = model(feat)
                loss = criterion(logits, label)
                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total += label.numel()

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_acc = correct / total if total else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}/{pred_cfg['epochs']} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    # Final test-set evaluation using best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for feat, label in test_loader:
            feat = feat.to(device)
            logits = model(feat)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(label.numpy())

    if all_preds:
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_targets = np.concatenate([t.flatten() for t in all_targets])
        metrics = compute_metrics(all_preds, all_targets, pred_cfg["num_classes"])
    else:
        metrics = {}

    outputs_dir = os.path.join(ROOT, config["paths"]["maps_dir"])
    os.makedirs(outputs_dir, exist_ok=True)
    with open(os.path.join(outputs_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(outputs_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Test Set Metrics ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"\nBest model saved to {best_path}")
    return model, history, metrics


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
