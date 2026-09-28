"""Central configuration.

Every number the audio front-end depends on lives here, because training and
inference must agree exactly. A mismatch of a single hop length silently
degrades the deployed model without raising anything.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAVDESS_DIR = DATA_DIR / "ravdess"
CKPT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports"

# --- audio front-end (slide 9: 16 kHz mono, 3 s, 25 ms / 10 ms, 64 mel bands) ---
SAMPLE_RATE = 16_000
CLIP_SECONDS = 3.0
N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)   # 48000
N_FFT = 400                                   # 25 ms
HOP_LENGTH = 160                              # 10 ms
WIN_LENGTH = 400
N_MELS = 64
F_MIN = 20.0
F_MAX = 8000.0
TOP_DB = 80.0
N_FRAMES = N_SAMPLES // HOP_LENGTH + 1        # 301

# --- labels ---
# RAVDESS emotion code (3rd field of the filename) -> label name
RAVDESS_EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
# Order is frozen: the classifier's output index i means EMOTIONS[i] everywhere,
# including in the deployed app. Never reorder without retraining.
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
N_CLASSES = len(EMOTIONS)

# --- speaker-independent split (slide 7: no speaker in two splits) ---
# RAVDESS actors are 1..24, odd = male, even = female, so these three groups
# are each sex-balanced.
TRAIN_ACTORS = list(range(1, 17))    # 16 actors, 960 clips
VAL_ACTORS = [17, 18, 19, 20]        # 4 actors, 240 clips -- early stopping only
TEST_ACTORS = [21, 22, 23, 24]       # 4 actors, 240 clips -- touched once, at the end

SEED = 1337
