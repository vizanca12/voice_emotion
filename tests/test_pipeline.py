"""Invariants that must hold for the reported numbers to mean anything.

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C
from src.data import build_index, make_loaders, parse_filename, speaker_folds
from src.features import LogMel, fix_length
from src.models import ScratchCNN, TransferResNet, count_parameters


# --------------------------------------------------------------- front-end


def test_logmel_shape_matches_config():
    spec = LogMel()(torch.randn(C.N_SAMPLES))
    assert spec.shape == (1, 1, C.N_MELS, C.N_FRAMES) == (1, 1, 64, 301)


def test_logmel_is_standardised():
    spec = LogMel()(torch.randn(C.N_SAMPLES))
    assert abs(float(spec.mean())) < 1e-4
    assert abs(float(spec.std()) - 1.0) < 1e-2


def test_logmel_is_gain_invariant():
    """A quiet recording and a loud one must reach the network alike -- a laptop
    mic is nowhere near the corpus's recording level."""
    wav = torch.randn(C.N_SAMPLES)
    quiet = LogMel()(wav * 0.01)
    loud = LogMel()(wav * 1.0)
    assert torch.allclose(quiet, loud, atol=1e-3)


@pytest.mark.parametrize("n", [1000, C.N_SAMPLES - 1, C.N_SAMPLES, C.N_SAMPLES + 1, 200_000])
def test_fix_length_always_returns_exact_window(n):
    assert fix_length(torch.randn(n)).numel() == C.N_SAMPLES


def test_fix_length_picks_the_loudest_window():
    """Silence, then a burst: the burst must survive the crop."""
    wav = torch.zeros(C.N_SAMPLES * 3)
    burst_start = C.N_SAMPLES * 2
    wav[burst_start:burst_start + 8000] = 1.0
    cropped = fix_length(wav)
    assert cropped.abs().sum() > 7000, "the energetic part was cropped away"


# ------------------------------------------------------------------ splits


def test_configured_splits_are_speaker_disjoint():
    train, val, test = set(C.TRAIN_ACTORS), set(C.VAL_ACTORS), set(C.TEST_ACTORS)
    assert not train & val and not train & test and not val & test
    assert train | val | test == set(range(1, 25)), "every actor must be used exactly once"


@pytest.mark.parametrize("n_folds", [3, 5, 6])
def test_cv_folds_are_speaker_disjoint(n_folds):
    for fold in speaker_folds(n_folds):
        train, val, test = set(fold["train"]), set(fold["val"]), set(fold["test"])
        assert not train & val and not train & test and not val & test
        assert train and val and test


def test_cv_covers_every_actor_exactly_once_as_test():
    seen = [a for fold in speaker_folds(5) for a in fold["test"]]
    assert sorted(seen) == list(range(1, 25))


def test_make_loaders_rejects_a_speaker_leak():
    """The guard that stops the single most common mistake in this task."""
    cache = {
        "specs": torch.zeros(4, 1, C.N_MELS, C.N_FRAMES),
        "labels": torch.zeros(4, dtype=torch.long),
        "actors": torch.tensor([1, 2, 3, 4]),
    }
    with pytest.raises(AssertionError, match="speaker leak"):
        make_loaders(cache, train_actors=[1, 2], val_actors=[2, 3], test_actors=[4])


# ------------------------------------------------------------------ corpus


def test_filename_parser_reads_every_field():
    clip = parse_filename(Path("03-01-06-01-02-01-12.wav"))
    assert clip.emotion == "fearful"
    assert clip.actor == 12
    assert clip.sex == "female"          # even actor ids are female in RAVDESS
    assert clip.intensity == "normal"


def test_filename_parser_skips_song_and_video():
    assert parse_filename(Path("03-02-06-01-02-01-12.wav")) is None   # song channel
    assert parse_filename(Path("02-01-06-01-02-01-12.wav")) is None   # video-only


@pytest.mark.skipif(not C.RAVDESS_DIR.exists(), reason="corpus not downloaded")
def test_corpus_is_complete():
    clips = build_index()
    assert len(clips) == 1440
    assert len(set(c.actor for c in clips)) == 24


# ------------------------------------------------------------------ models


def test_scratch_cnn_size_matches_the_brief():
    total, _ = count_parameters(ScratchCNN())
    assert 1.0e6 < total < 1.4e6, f"expected ~1.2 M parameters, got {total/1e6:.2f} M"


def test_models_accept_the_front_end_output():
    x = torch.randn(2, 1, C.N_MELS, C.N_FRAMES)
    assert ScratchCNN()(x).shape == (2, C.N_CLASSES)
    assert TransferResNet(pretrained=False)(x).shape == (2, C.N_CLASSES)


def test_frozen_backbone_trains_only_the_head():
    model = TransferResNet(pretrained=False, freeze=True, unfreeze_layer4=False)
    _, trainable = count_parameters(model)
    assert trainable == 512 * C.N_CLASSES + C.N_CLASSES


def test_frozen_batchnorm_stays_in_eval_mode():
    """Otherwise the running statistics drift while the head is being fitted."""
    model = TransferResNet(pretrained=False, freeze=True).train()
    bns = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)


# ---------------------------------------------------- train / serve parity


def test_app_and_training_compute_identical_features():
    """The single most damaging bug this project can have.

    If the server's front-end drifts from the training one, the deployed model
    is fed a distribution it never saw. Nothing raises; the demo just gets
    worse. Both paths import src.features, and this pins that down.
    """
    from app.main import _frontend

    wav = torch.randn(50_000)
    training_side = LogMel()(fix_length(wav))
    serving_side = _frontend(fix_length(wav))
    assert torch.allclose(training_side, serving_side, atol=1e-6)
