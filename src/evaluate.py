"""Metrics and reporting.

The headline number is UAR (unweighted average recall = macro recall), not
accuracy. RAVDESS has half as many neutral clips as any other emotion, so a
model that never predicts neutral still scores well on accuracy. UAR is the
standard metric in the speech-emotion literature for exactly this reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import config as C


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        targets.append(y)
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int = C.N_CLASSES) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(logits: np.ndarray, y_true: np.ndarray) -> dict:
    y_pred = logits.argmax(1)
    cm = confusion(y_true, y_pred)
    support = cm.sum(1)
    seen = support > 0
    recall = np.zeros(C.N_CLASSES)
    recall[seen] = np.diag(cm)[seen] / support[seen]
    return {
        "uar": float(recall[seen].mean()),            # macro recall -- the headline
        "accuracy": float((y_pred == y_true).mean()),
        "per_class_recall": {C.EMOTIONS[i]: float(recall[i]) for i in range(C.N_CLASSES) if seen[i]},
        "support": {C.EMOTIONS[i]: int(support[i]) for i in range(C.N_CLASSES) if seen[i]},
        "confusion": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    """Row-normalised confusion matrix, as on slide 21."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = cm.sum(1, keepdims=True)
    keep = rows.squeeze(1) > 0
    norm = np.divide(cm, np.maximum(rows, 1)) * 100.0
    labels = [e for e, k in zip(C.EMOTIONS, keep) if k]
    norm = norm[np.ix_(keep, keep)]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = norm[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v < 55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_report(metrics: dict, name: str, report_dir: Path = C.REPORT_DIR) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion(np.array(metrics["confusion"]), report_dir / f"{name}_confusion.png",
                   f"{name} -- unseen speakers, row-normalised %")


def format_table(metrics: dict) -> str:
    lines = [f"  UAR {metrics['uar']*100:5.1f} %   accuracy {metrics['accuracy']*100:5.1f} %"
             f"   (n={metrics['n_samples']})", "  per-class recall:"]
    for emo, r in metrics["per_class_recall"].items():
        lines.append(f"    {emo:<10s} {r*100:5.1f} %   (n={metrics['support'][emo]})")
    return "\n".join(lines)
