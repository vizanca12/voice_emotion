"""Run the trained models on audio files from the command line.

    python scripts/predict.py recording.wav
    python scripts/predict.py data/ravdess/Actor_21            # a whole folder
    python scripts/predict.py data/ravdess/Actor_2[1-4] --model scratch

When a filename follows the RAVDESS convention the true emotion is read from
it, so the output doubles as a spot check: the summary line at the end is the
accuracy over the files you passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C                             # noqa: E402
from src.data import parse_filename                     # noqa: E402
from src.features import LogMel, fix_length, load_wav   # noqa: E402
from src.models import build_model                      # noqa: E402


def load_checkpoints(only: list[str] | None) -> dict[str, torch.nn.Module]:
    models = {}
    for path in sorted(C.CKPT_DIR.glob("*.pt")):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        name = blob["model_name"]
        if only and name not in only:
            continue
        kwargs = {"pretrained": False} if name.startswith("transfer") else {}
        model = build_model(name, **kwargs)
        model.load_state_dict(blob["state_dict"])
        models[name] = model.eval()
    if not models:
        raise SystemExit("no checkpoints found -- run `python -m src.train --all` first")
    return models


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        files.extend(sorted(p.rglob("*.wav")) if p.is_dir() else [p])
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"not found: {missing[0]}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="wav files or folders")
    ap.add_argument("--model", action="append", dest="models",
                    help="restrict to one model (repeatable)")
    ap.add_argument("--top", type=int, default=3, help="how many emotions to show")
    args = ap.parse_args()

    models = load_checkpoints(args.models)
    frontend = LogMel().eval()
    files = collect(args.paths)
    correct = {name: 0 for name in models}
    n_labelled = 0

    for path in files:
        clip = parse_filename(path)
        truth = clip.emotion if clip else None
        with torch.no_grad():
            spec = frontend(fix_length(load_wav(path)))

        print(f"\n{path.name}" + (f"   [true: {truth}]" if truth else ""))
        if truth:
            n_labelled += 1
        for name, model in models.items():
            with torch.no_grad():
                probs = F.softmax(model(spec), dim=1)[0]
            order = torch.argsort(probs, descending=True)[: args.top]
            top = C.EMOTIONS[int(order[0])]
            mark = "" if truth is None else ("  OK" if top == truth else "  --")
            shown = "  ".join(f"{C.EMOTIONS[i]} {probs[i]*100:.0f}%" for i in order)
            print(f"  {name:<13} {shown}{mark}")
            if truth and top == truth:
                correct[name] += 1

    if n_labelled > 1:
        print(f"\naccuracy over {n_labelled} labelled files")
        for name in models:
            print(f"  {name:<13} {correct[name]/n_labelled*100:5.1f} %  "
                  f"({correct[name]}/{n_labelled})")
        print("\nNote: on RAVDESS actors 1-20 this is a training-set score and means "
              "nothing.\nOnly actors 21-24 were held out.")


if __name__ == "__main__":
    main()
