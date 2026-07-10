"""Regression tests for cache invalidation and safe training restarts."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import prepare_data
from train import (
    atomic_copy,
    remove_restart_artifacts,
    restore_rng,
    should_update_best_rankic,
    train_epoch,
)


def test_config_hash_covers_nested_yaml_changes():
    base = {
        "qlib_init": {"provider_uri": "data", "region": "cn"},
        "data_handler_config": {"label": ["future_return"]},
        "task": {"dataset": {"kwargs": {"segments": {"train": [1, 2]}, "step_len": 8}}},
    }
    changed = {
        **base,
        "task": {"dataset": {"kwargs": {"segments": {"train": [1, 3]}, "step_len": 8}}},
    }
    assert prepare_data.config_hash(base) != prepare_data.config_hash(changed)


def test_dataset_cache_requires_matching_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "ROOT", str(tmp_path))
    monkeypatch.setattr(
        prepare_data,
        "dataset_path",
        lambda universe, tag, split: str(
            tmp_path / "dataset" / universe / f"{universe}_dl_{split}_{tag}.pkl"
        ),
    )
    universe, tag, fingerprint = "csi300", "test", "abc123"
    split_dir = tmp_path / "dataset" / universe
    split_dir.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (split_dir / f"{universe}_dl_{split}_{tag}.pkl").write_bytes(b"data")

    assert not prepare_data.dataset_cache_matches(universe, tag, fingerprint)
    prepare_data.write_dataset_manifest(universe, tag, fingerprint)
    assert prepare_data.dataset_cache_matches(universe, tag, fingerprint)

    manifest = prepare_data.dataset_manifest_path(universe, tag)
    with open(manifest, "r") as f:
        payload = json.load(f)
    payload["config_hash"] = "stale"
    with open(manifest, "w") as f:
        json.dump(payload, f)
    assert not prepare_data.dataset_cache_matches(universe, tag, fingerprint)


def test_restart_removes_final_and_atomic_copy_publishes(tmp_path):
    resume = tmp_path / "last.pth"
    best = tmp_path / "best.pth"
    final = tmp_path / "TEST.pth"
    for path in (resume, best, final):
        path.write_bytes(path.name.encode())

    remove_restart_artifacts(str(resume), str(best), str(final))
    assert not resume.exists()
    assert not best.exists()
    assert not final.exists()

    source = tmp_path / "new_best.pth"
    source.write_bytes(b"complete checkpoint")
    atomic_copy(str(source), str(final))
    assert final.read_bytes() == b"complete checkpoint"
    assert not (tmp_path / "TEST.pth.tmp").exists()


def test_train_loss_mask_does_not_filter_spatial_attention_inputs():
    class SpyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.seen_batch_sizes = []

        def forward(self, x):
            self.seen_batch_sizes.append(x.shape[0])
            return self.bias.expand(x.shape[0])

    class Scheduler:
        def step(self):
            pass

    # DataLoader convention: outer singleton batch, then N stocks x T x F.
    data = torch.zeros(1, 6, 8, 222)
    data[0, :, -1, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")])
    model = SpyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_epoch([data], optimizer, Scheduler(), model, torch.device("cpu"))

    # NaN/extreme labels are excluded from loss, not from the model cross-section.
    assert model.seen_batch_sizes == [6]


def test_restore_rng_accepts_checkpoint_tensor_state():
    original = torch.get_rng_state()
    torch.manual_seed(123)
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(456)
    restore_rng({
        "rng_torch": expected_state,
        "rng_cuda": None,
        "rng_np": __import__("numpy").random.get_state(),
        "rng_py": __import__("random").getstate(),
    })
    assert torch.equal(torch.get_rng_state(), expected_state)
    torch.set_rng_state(original)


def test_rankic_checkpoint_selection_maximizes_and_rejects_nan():
    assert should_update_best_rankic(0.03, 0.02)
    assert not should_update_best_rankic(0.01, 0.02)
    assert not should_update_best_rankic(0.02, 0.02)
    assert not should_update_best_rankic(float("nan"), 0.02)
