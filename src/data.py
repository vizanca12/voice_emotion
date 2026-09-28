"""RAVDESS indexing, speaker-disjoint splits, and a cached spectrogram dataset."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .features import LogMel, fix_length, load_wav

CACHE_PATH = C.DATA_DIR / "cache_logmel.pt"


@dataclasses.dataclass(frozen=True)
class Clip:
    path: Path
    emotion: str
    label: int
    actor: int
    intensity: str   # "normal" | "strong"
    statement: int
    repetition: int

    @property
    def sex(self) -> str:
        return "male" if self.actor % 2 == 1 else "female"


def parse_filename(path: Path) -> Clip | None:
    """RAVDESS encodes every attribute in the filename:
    modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-06-01-02-01-12.wav = audio-only, speech, fearful, normal, ... , actor 12
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion_code, intensity, statement, repetition, actor = parts
    if modality != "03" or channel != "01":      # audio-only speech
        return None
    emotion = C.RAVDESS_EMOTION_CODES.get(emotion_code)
    if emotion is None or emotion not in C.EMOTION_TO_IDX:
        return None
    return Clip(
        path=path,
        emotion=emotion,
        label=C.EMOTION_TO_IDX[emotion],
        actor=int(actor),
        intensity="strong" if intensity == "02" else "normal",
        statement=int(statement),
        repetition=int(repetition),
    )


def build_index(root: Path = C.RAVDESS_DIR) -> list[Clip]:
    clips = [c for p in sorted(root.rglob("*.wav")) if (c := parse_filename(p)) is not None]
    if not clips:
        raise FileNotFoundError(f"No RAVDESS wav files under {root}")
    return clips


# --------------------------------------------------------------------------- cache


def build_cache(force: bool = False) -> dict:
    """Compute every clip's log-mel once and store it.

    1440 clips at (64, 301) in float16 is ~55 MB, so the whole corpus lives in
    RAM and an epoch never touches the disk. Training a fold then takes seconds.
    """
    if CACHE_PATH.exists() and not force:
        return torch.load(CACHE_PATH, weights_only=False)

    clips = build_index()
    frontend = LogMel()
    specs = torch.empty(len(clips), 1, C.N_MELS, C.N_FRAMES, dtype=torch.float16)

    for i, clip in enumerate(clips):
        wav = fix_length(load_wav(clip.path))
        with torch.no_grad():
            specs[i] = frontend(wav).squeeze(0).half()
        if (i + 1) % 200 == 0:
            print(f"  featurised {i + 1}/{len(clips)}")

    cache = {
        "specs": specs,
        "labels": torch.tensor([c.label for c in clips], dtype=torch.long),
        "actors": torch.tensor([c.actor for c in clips], dtype=torch.long),
        "emotions": C.EMOTIONS,
        "paths": [str(c.path) for c in clips],
    }
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)
    print(f"cache -> {CACHE_PATH}  {specs.shape}  {CACHE_PATH.stat().st_size / 1e6:.0f} MB")
    return cache


# --------------------------------------------------------------------------- dataset


class SpecDataset(Dataset):
    def __init__(self, specs: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.specs, self.labels, self.augment = specs, labels, augment

    def __len__(self) -> int:
        return self.specs.shape[0]

    def __getitem__(self, i: int):
        spec = self.specs[i].float()
        if self.augment:
            from .features import spec_augment
            spec = spec_augment(spec)
        return spec, self.labels[i]


def subset(cache: dict, actors: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isin(cache["actors"], torch.tensor(actors))
    return cache["specs"][mask], cache["labels"][mask]


def make_loaders(cache: dict, train_actors, val_actors, test_actors, batch_size: int = 32):
    """Build the three loaders and assert the split really is speaker-disjoint.

    Slide 7 is the whole point of the exercise: if one actor leaks across splits
    the network can score high by recognising the person. The assert makes that
    mistake impossible to commit silently.
    """
    groups = [set(train_actors), set(val_actors), set(test_actors)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        overlap = groups[a] & groups[b]
        assert not overlap, f"speaker leak between splits: actors {sorted(overlap)}"

    tr_x, tr_y = subset(cache, train_actors)
    va_x, va_y = subset(cache, val_actors)
    te_x, te_y = subset(cache, test_actors)

    return (
        DataLoader(SpecDataset(tr_x, tr_y, augment=True), batch_size=batch_size,
                   shuffle=True, drop_last=True),
        DataLoader(SpecDataset(va_x, va_y), batch_size=batch_size),
        DataLoader(SpecDataset(te_x, te_y), batch_size=batch_size),
    )


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights. RAVDESS has 96 neutral clips against 192 of
    every other emotion, so an unweighted loss under-serves neutral."""
    counts = torch.bincount(labels, minlength=C.N_CLASSES).float().clamp(min=1)
    w = counts.sum() / (C.N_CLASSES * counts)
    return w


def speaker_folds(n_folds: int = 5, actors: list[int] | None = None) -> list[dict]:
    """Partition actors into n_folds speaker-disjoint folds for grouped CV.

    Actors alternate male/female by parity, so striding by n_folds keeps every
    fold roughly sex-balanced.
    """
    actors = sorted(actors or list(range(1, 25)))
    folds = []
    for k in range(n_folds):
        test = actors[k::n_folds]
        rest = [a for a in actors if a not in test]
        val = rest[::4]                       # a quarter of the rest, for early stopping
        train = [a for a in rest if a not in val]
        folds.append({"train": train, "val": val, "test": test})
    return folds
