import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prediction.unet_model import FireRiskUNet, count_parameters
from prediction.convlstm_model import FireDangerConvLSTM
from prediction.train_unet import compute_metrics, PatchDataset, split_indices


def test_unet_output_shape():
    model = FireRiskUNet(in_channels=13, num_classes=4, base_filters=16, depth=3)
    x = torch.randn(2, 13, 128, 128)
    out = model(x)
    assert out.shape == (2, 4, 128, 128)


def test_unet_handles_non_power_of_two_input():
    """U-NET with skip connections must handle odd input sizes via padding in Up blocks."""
    model = FireRiskUNet(in_channels=13, num_classes=4, base_filters=16, depth=3)
    x = torch.randn(1, 13, 100, 100)
    out = model(x)
    assert out.shape == (1, 4, 100, 100)


def test_unet_predict_risk_map():
    model = FireRiskUNet(in_channels=13, num_classes=4, base_filters=16, depth=2)
    x = torch.randn(1, 13, 64, 64)
    classes, probs = model.predict_risk_map(x)
    assert classes.shape == (1, 64, 64)
    assert probs.shape == (1, 4, 64, 64)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 64, 64), atol=1e-4)
    assert classes.min() >= 0 and classes.max() <= 3


def test_unet_parameter_count_positive():
    model = FireRiskUNet(in_channels=13, num_classes=4, base_filters=32, depth=4)
    assert count_parameters(model) > 0


def test_convlstm_output_shape():
    model = FireDangerConvLSTM(in_channels=6, hidden_dim=8)
    x = torch.randn(2, 5, 6, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)
    assert (out >= 0).all() and (out <= 1).all(), "ConvLSTM danger output must be in [0,1] (sigmoid readout)"


def test_compute_metrics_perfect_prediction():
    preds = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    targets = preds.copy()
    metrics = compute_metrics(preds, targets, num_classes=4)
    assert metrics["accuracy"] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["kappa"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_worst_case():
    preds = np.array([0, 0, 0, 0])
    targets = np.array([1, 1, 1, 1])
    metrics = compute_metrics(preds, targets, num_classes=4)
    assert metrics["accuracy"] == 0.0


def test_patch_dataset_extracts_correct_shapes():
    c, h, w = 13, 100, 100
    features = np.random.randn(c, h, w).astype(np.float32)
    labels = np.random.randint(0, 4, size=(h, w)).astype(np.uint8)
    ds = PatchDataset(features, labels, patch_size=32, stride=16)
    assert len(ds) > 0
    feat_patch, label_patch = ds[0]
    assert feat_patch.shape == (13, 32, 32)
    assert label_patch.shape == (32, 32)


def test_split_indices_no_overlap_and_correct_sizes():
    n = 100
    train_idx, val_idx, test_idx = split_indices(n, val_split=0.15, test_split=0.15, seed=42)
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    assert len(val_idx) == 15
    assert len(test_idx) == 15
