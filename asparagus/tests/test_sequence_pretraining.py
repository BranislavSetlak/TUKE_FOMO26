import pytest
import torch
from asparagus.functional.sequence_labels import (
    effective_number_class_weights,
    sequence_class_counts,
    sequence_class_id,
)
from asparagus.modules.datasets.PretrainDataset import PretrainDataset
from asparagus.modules.lightning_modules import SelfSupervisedModule
from torch import nn


RAW_TO_CLASS = {"T1w": 0, "T2w": 1, "FLAIR": 2}
IGNORED = {"scan": -100}


class DummyHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_with_features(self, x):
        features = x.mean(dim=(-3, -2, -1), keepdim=True) * self.scale
        logits = torch.cat([features.flatten(1), -features.flatten(1), features.flatten(1) * 0], dim=1)
        return x * self.scale, features, logits


def test_sequence_mapping_counts_and_effective_weights():
    paths = [
        "/data/sub-1/anat/sub-1_T1w.pt",
        "/data/sub-2/anat/sub-2_T1w.pt",
        "/data/sub-3/anat/sub-3_T2w.pt",
        "/data/sub-4/anat/sub-4_FLAIR.pt",
        "/data/sub-5/anat/sub-5_scan.pt",
        "/data/sub-6/anat/sub-6_newsequence.pt",
    ]
    counts = sequence_class_counts(
        paths,
        raw_to_class=RAW_TO_CLASS,
        ignored_sequences=IGNORED,
        other_class_id=2,
        num_classes=3,
    )
    assert counts == [2, 1, 2]
    weights = effective_number_class_weights(counts, beta=0.9)
    assert sum(weights) == pytest.approx(3.0)
    assert weights[1] > weights[0]
    assert sequence_class_id(paths[4], RAW_TO_CLASS, IGNORED, other_class_id=2) == -100


def test_pretrain_dataset_adds_sequence_target(pretrain_files):
    dataset = PretrainDataset(
        pretrain_files["train"],
        sequence_raw_to_class=RAW_TO_CLASS,
        sequence_ignored=IGNORED,
        sequence_other_class_id=2,
    )
    item = dataset[0]
    assert item["sequence_label"].dtype == torch.long
    assert item["sequence_label"].item() == 2


def test_all_ignored_sequence_batch_has_finite_zero_loss():
    module = SelfSupervisedModule(
        model=DummyHybrid(),
        warmup_epochs=0,
        compile_mode=None,
        sequence_loss_weight=0.01,
        sequence_class_weights=[1.0, 1.0, 1.0],
        sequence_ignore_index=-100,
    )
    logits = torch.randn(2, 3, requires_grad=True)
    loss, metrics = module._sequence_loss(logits, torch.tensor([-100, -100]))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)
    assert metrics["valid_fraction"].item() == pytest.approx(0.0)


def test_sequence_weights_do_not_cancel_for_small_batches():
    module = SelfSupervisedModule(
        model=DummyHybrid(),
        warmup_epochs=0,
        compile_mode=None,
        sequence_loss_weight=0.01,
        sequence_class_weights=[2.0, 1.0, 1.0],
        sequence_ignore_index=-100,
    )
    logits = torch.zeros(1, 3, requires_grad=True)
    loss, _ = module._sequence_loss(logits, torch.tensor([0]))
    assert loss.item() == pytest.approx(2.0 * torch.log(torch.tensor(3.0)).item())
