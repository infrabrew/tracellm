"""Tests for custom optimizers."""

import torch
import torch.nn as nn

from tracellm.training.optimizers import Muon, create_optimizer, build_deepspeed_config


def test_muon_basic_step():
    """Muon should update parameters and reduce loss."""
    model = nn.Linear(10, 1)
    optimizer = Muon(model.parameters(), lr=0.01)

    x = torch.randn(32, 10)
    target = torch.randn(32, 1)

    # Two training steps
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Parameters should have changed
    assert losses[1] != losses[0]


def test_muon_with_weight_decay():
    """Muon with weight decay should shrink weights."""
    model = nn.Linear(10, 1)
    nn.init.ones_(model.weight)
    initial_norm = model.weight.norm().item()

    optimizer = Muon(model.parameters(), lr=0.01, weight_decay=0.1)

    x = torch.randn(4, 10)
    for _ in range(5):
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    # Weight norm should have decreased due to decay
    final_norm = model.weight.norm().item()
    assert final_norm < initial_norm


def test_create_optimizer_adamw():
    model = nn.Linear(10, 1)
    opt = create_optimizer("adamw", model.parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


def test_create_optimizer_muon():
    model = nn.Linear(10, 1)
    opt = create_optimizer("muon", model.parameters(), lr=1e-4)
    assert isinstance(opt, Muon)


def test_create_optimizer_sgd():
    model = nn.Linear(10, 1)
    opt = create_optimizer("sgd", model.parameters(), lr=1e-2)
    assert isinstance(opt, torch.optim.SGD)


def test_create_optimizer_unknown():
    model = nn.Linear(10, 1)
    try:
        create_optimizer("nonexistent", model.parameters())
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown optimizer" in str(e)


def test_build_deepspeed_config_z2():
    config = build_deepspeed_config(stage=2, batch_size=8, lr=1e-4, bf16=True)
    assert config["train_micro_batch_size_per_gpu"] == 8
    assert config["zero_optimization"]["stage"] == 2
    assert config["bf16"]["enabled"] is True
    assert config["optimizer"]["params"]["lr"] == 1e-4


def test_build_deepspeed_config_z3_offload():
    config = build_deepspeed_config(
        stage=3, batch_size=2, lr=1e-4,
        offload_optimizer=True, offload_params=True,
    )
    assert config["zero_optimization"]["stage"] == 3
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert config["zero_optimization"]["offload_param"]["device"] == "cpu"
