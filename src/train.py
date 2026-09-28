"""Training entry point.

    python -m src.train --model scratch
    python -m src.train --all
    python -m src.train --model scratch --cv 5     # speaker-grouped cross-validation

The test split is evaluated exactly once, after early stopping has already
chosen the checkpoint on a *separate* set of speakers. Selecting the epoch on
the test set would leak it, and the reported number would no longer mean
"performance on speakers the model never heard".
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import build_cache, class_weights, make_loaders, speaker_folds
from .evaluate import compute_metrics, format_table, predict, save_report
from .models import MODEL_NAMES, build_model, count_parameters


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = C.SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(
    model_name: str,
    cache: dict,
    split: dict,
    epochs: int = 60,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    device = device or get_device()
    set_seed()

    train_loader, val_loader, test_loader = make_loaders(
        cache, split["train"], split["val"], split["test"], batch_size
    )
    model = build_model(model_name).to(device)
    total, trainable = count_parameters(model)

    weights = class_weights(train_loader.dataset.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.25
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_uar, best_state, best_epoch = -1.0, None, -1
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)
        val = compute_metrics(*predict(model, val_loader, device))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_uar": val["uar"], "val_acc": val["accuracy"]})

        if val["uar"] > best_uar:
            best_uar, best_epoch = val["uar"], epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{epochs}  loss {train_loss:.3f}  "
                  f"val UAR {val['uar']*100:5.1f} %  (best {best_uar*100:5.1f} % @ {best_epoch})")

    model.load_state_dict(best_state)
    test = compute_metrics(*predict(model, test_loader, device))
    elapsed = time.time() - t0

    if verbose:
        print(f"  best val UAR {best_uar*100:.1f} % at epoch {best_epoch}; "
              f"trained in {elapsed:.0f} s on {device.type}")
        print(f"  TEST (speakers {split['test']}):")
        print(format_table(test))

    return {
        "model": model_name,
        "state_dict": best_state,
        "test": test,
        "val_uar": best_uar,
        "best_epoch": best_epoch,
        "history": history,
        "params_total": total,
        "params_trainable": trainable,
        "split": split,
        "seconds": elapsed,
    }


def save_checkpoint(result: dict, tag: str) -> None:
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = C.CKPT_DIR / f"{tag}.pt"
    torch.save(
        {
            "state_dict": result["state_dict"],
            "model_name": result["model"],
            "emotions": C.EMOTIONS,
            "test_metrics": {k: v for k, v in result["test"].items() if k != "confusion"},
            "audio": {"sample_rate": C.SAMPLE_RATE, "n_mels": C.N_MELS,
                      "n_fft": C.N_FFT, "hop_length": C.HOP_LENGTH,
                      "clip_seconds": C.CLIP_SECONDS},
        },
        path,
    )
    print(f"  checkpoint -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def run_cv(model_name: str, cache: dict, n_folds: int, epochs: int, lr: float) -> dict:
    """Speaker-grouped cross-validation: every one of the 24 actors is unseen exactly once."""
    device = get_device()
    uars, accs = [], []
    for k, split in enumerate(speaker_folds(n_folds), 1):
        print(f"\n fold {k}/{n_folds}  test actors {split['test']}")
        r = train_one(model_name, cache, split, epochs=epochs, lr=lr,
                      device=device, verbose=False)
        uars.append(r["test"]["uar"])
        accs.append(r["test"]["accuracy"])
        print(f"  fold UAR {r['test']['uar']*100:5.1f} %   acc {r['test']['accuracy']*100:5.1f} %")
    summary = {
        "model": model_name, "n_folds": n_folds,
        "uar_mean": float(np.mean(uars)), "uar_std": float(np.std(uars)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "fold_uars": uars,
    }
    print(f"\n{model_name}: UAR {summary['uar_mean']*100:.1f} +/- {summary['uar_std']*100:.1f} % "
          f"over {n_folds} speaker-disjoint folds")
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORT_DIR / f"cv_{model_name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES, default="scratch")
    ap.add_argument("--all", action="store_true", help="train every model in turn")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cv", type=int, default=0, help="speaker-grouped folds (0 = fixed split)")
    args = ap.parse_args()

    # Sensible per-model defaults: the frozen transfer head converges much
    # faster than a network learning filters from noise.
    defaults = {"scratch": (60, 3e-4), "transfer": (30, 1e-3), "transfer_ft": (30, 3e-4)}

    print(f"device: {get_device()}")
    print("building / loading spectrogram cache ...")
    cache = build_cache()
    print(f"  {cache['specs'].shape[0]} clips, {len(set(cache['actors'].tolist()))} actors")

    models = MODEL_NAMES if args.all else [args.model]
    split = {"train": C.TRAIN_ACTORS, "val": C.VAL_ACTORS, "test": C.TEST_ACTORS}

    for name in models:
        epochs = args.epochs or defaults[name][0]
        lr = args.lr or defaults[name][1]
        print(f"\n=== {name}  ({epochs} epochs, lr {lr}) ===")
        if args.cv:
            run_cv(name, cache, args.cv, epochs, lr)
        else:
            result = train_one(name, cache, split, epochs=epochs, lr=lr,
                               batch_size=args.batch_size)
            save_checkpoint(result, name)
            save_report(result["test"], name)


if __name__ == "__main__":
    main()
