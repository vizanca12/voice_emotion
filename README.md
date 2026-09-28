---
title: Voice Emotion Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Can a machine hear how you feel?

Speech emotion recognition with convolutional neural networks.
Record a sentence in the browser, watch the spectrogram build in real time, and
see what three CNNs hear in your voice.

- **Live app:** <!-- TODO: https://huggingface.co/spaces/USER/voice-emotion-cnn -->
- **Repository:** <!-- TODO: https://github.com/USER/voice-emotion-cnn -->

Course assignment, *Apprentissage pour la robotique* (ENSTA Paris, 3A) — due 29/09/2026.

---

## Results

Trained on [RAVDESS](https://zenodo.org/records/1188976): 1440 clips, 24 actors,
8 emotions. **Every number below is on four actors the models never heard.**

| Model | Architecture | Parameters (trainable) | UAR | Accuracy |
|---|---|---|---|---|
| **`scratch`** | 4 conv blocks, GAP + linear | 1.17 M (all) | **64.1 %** | 63.7 % |
| `transfer_ft` | ResNet-18 (ImageNet), `layer4` fine-tuned | 11.2 M (8.4 M) | 59.0 % | 59.6 % |
| `transfer` | ResNet-18 (ImageNet), backbone frozen | 11.2 M (4.1 k) | 34.4 % | 36.7 % |

Chance on 8 classes is 12.5 %. For scale, a published CNN reporting 99.89 % on a
two-speaker corpus was measuring speaker identity, not emotion.

Under 5-fold speaker-grouped cross-validation — every actor held out exactly
once — `scratch` scores **65.8 % ± 5.1 % UAR**, so the single-split number above
is representative rather than a lucky draw. Details
[below](#cross-validation-is-641--a-lucky-draw).

### The headline metric is UAR, not accuracy

RAVDESS records neutral at one intensity and every other emotion at two, so
there are 96 neutral clips against 192 of each other class. A model that simply
never predicts neutral loses little accuracy. **UAR** — unweighted average
recall, the mean of the per-class recalls — charges it the full price, and is
the standard metric in the speech-emotion literature for that reason.

### Per-class recall (`scratch`)

| | neutral | calm | happy | sad | angry | fearful | disgust | surprised |
|---|---|---|---|---|---|---|---|---|
| recall | 68.8 % | 81.2 % | 50.0 % | **25.0 %** | 68.8 % | 56.2 % | 68.8 % | 93.8 % |

![confusion matrix](reports/scratch_confusion.png)

**The errors follow arousal, not valence.** Low-energy emotions collapse into
each other — 28 % of sad is heard as calm, another 28 % as disgust — while
high-energy ones do the same: 25 % of happy and 25 % of angry are heard as
surprised. The network reads how activated the voice is far more reliably than
whether the feeling is positive or negative. Sadness is the weakest class at
25 % recall, which is consistent: a quiet, slow, flat voice is acoustically
close to a calm one, and pitch contour alone does not separate them.

---

## Honest evaluation: how the split is made

This is the part that decides whether any of the numbers above mean anything.

Split the 1440 clips at random and the same actor lands in training and test.
The network can then score very highly by recognising *the person* — their
pitch range, their timbre, their recording conditions — and never learn
emotion at all. RAVDESS has only 24 speakers, so this leak is severe.

**Our protocol — every actor is used in exactly one split:**

| Split | Actors | Clips | Used for |
|---|---|---|---|
| train | 1–16 | 960 | fitting weights |
| validation | 17–20 | 240 | choosing the stopping epoch, nothing else |
| test | 21–24 | 240 | evaluated once, after training finished |

Actor ids alternate male/female in RAVDESS, so each group is sex-balanced.
The validation set is a *separate* group of speakers from test: picking the
best epoch on the test set would leak it just as surely as sharing an actor,
and the reported number would no longer describe unseen speakers.

`src/data.py` refuses to build loaders from overlapping actor groups:

```python
for a, b in ((0, 1), (0, 2), (1, 2)):
    overlap = groups[a] & groups[b]
    assert not overlap, f"speaker leak between splits: actors {sorted(overlap)}"
```

and `tests/test_pipeline.py` asserts that this guard fires.

### Cross-validation: is 64.1 % a lucky draw?

A single held-out group of four actors is a small sample. To check that the
result does not depend on *which* speakers were held out, `scratch` was
retrained under 5-fold speaker-grouped cross-validation, where each of the 24
actors is unseen exactly once:

```bash
python -m src.train --model scratch --cv 5
```

| Fold | 1 | 2 | 3 | 4 | 5 | |
|---|---|---|---|---|---|---|
| test actors | 1, 6, 11, 16, 21 | 2, 7, 12, 17, 22 | 3, 8, 13, 18, 23 | 4, 9, 14, 19, 24 | 5, 10, 15, 20 | |
| UAR | 67.5 % | 73.8 % | 65.9 % | 58.1 % | 63.7 % | **65.8 % ± 5.1 %** |

The fixed split's 64.1 % sits well inside one standard deviation of the mean,
so it was not a fortunate choice of actors.

The spread is worth reporting in its own right: **±5.1 points purely from which
speakers are held out**, on a 24-speaker corpus. A paper quoting a single split
to one decimal place is quoting precision it does not have, and two methods
differing by three points on one split cannot be separated at all.

---

## What transfer learning actually did

The frozen ImageNet backbone reached **34.4 % UAR — barely half** of the small
network trained from scratch, despite having ten times the parameters.
Unfreezing only `layer4` recovered **+24.6 points**, to 59.0 %.

This is the domain gap in plain numbers. ImageNet features describe object
textures and shapes under translation invariance in both axes. A spectrogram is
not that kind of image: translation along time is meaningful invariance, but
translation along *frequency* changes the pitch and therefore the class. The
early ImageNet filters (edges, blobs) do transfer — which is why fine-tuning
one stage is enough to close most of the gap — but the late, highly
object-specific layers do not, and freezing them forces the classifier to work
from features that were never meant for this signal.

The practical conclusion for this dataset: with 960 training clips, a 1.2 M
parameter network designed for the input beats an 11 M parameter network
designed for something else.

---

## Method

### From sound to an image

Both networks see the same 64 × 301 log-mel image, computed identically at
training time and inside the server — `src/features.py` is imported by both,
and a test pins the two paths together.

| Step | Value |
|---|---|
| resample, downmix | 16 kHz mono |
| window | most energetic 3.0 s (zero-padded if shorter) |
| STFT | 400-sample frame (25 ms), 160-sample hop (10 ms) |
| mel filterbank | 64 bands, 20 Hz – 8 kHz, power, converted to dB (80 dB floor) |
| normalisation | per-example: subtract mean, divide by standard deviation |

Choosing the loudest 3-second window rather than the first drops the leading
silence RAVDESS clips carry. Per-example standardisation makes the front end
gain-invariant, which matters a lot in deployment: a laptop microphone records
nowhere near the level of a studio corpus.

### Augmentation

SpecAugment (up to two frequency masks of 8 bands and two time masks of 30
frames) plus a random time shift of up to 0.4 s, applied on the spectrogram
during training only.

### Architectures

**`scratch`** — four blocks of `conv3×3 → BN → ReLU` twice, then `maxpool 2×2`,
at 32 → 64 → 128 → 256 channels; global average pooling; dropout; one linear
layer. 1.17 M parameters.

**`transfer` / `transfer_ft`** — ResNet-18 with ImageNet weights. The mel image
is repeated across three channels and resized to 224 × 224. `transfer` trains
only a new linear head; `transfer_ft` additionally unfreezes `layer4`. Frozen
BatchNorm layers are kept in eval mode, so their running statistics cannot
drift out from under the head while it is being fitted.

### Training

AdamW, OneCycle schedule, class-weighted cross-entropy with 0.05 label
smoothing, batch 32, mixed precision. The checkpoint kept is the epoch with the
best *validation* UAR. Training `scratch` takes about 80 seconds on an
RTX 3060.

---

## The app

Two independent audio paths run in the browser, which is what makes the live
view possible:

1. An `AnalyserNode` drives the **scrolling spectrogram while you speak**, with
   its rows mapped on the mel scale so the live view matches the model's axes.
2. A `MediaRecorder` captures the clip; it is decoded, resampled to 16 kHz and
   posted as raw float32.

Resampling in the browser means the server never decodes WebM/Opus, so the
container needs no ffmpeg — which is what keeps the deployed image small enough
for a free CPU Space.

The spectrogram shown after analysis is **the exact tensor the CNNs received**,
quantised to uint8 and base64-encoded (19 kB instead of 100 kB of JSON), not a
separate visualisation drawn for display.

| Endpoint | Purpose |
|---|---|
| `GET /` | the recording page |
| `GET /api/meta` | class names, loaded models, their test scores, audio settings |
| `POST /api/predict` | raw float32 PCM in the body, `?sr=` |
| `POST /api/predict-file` | multipart wav/flac/ogg upload |
| `GET /health` | liveness + which checkpoints loaded |

```bash
curl -F "file=@data/ravdess/Actor_21/03-01-05-01-01-01-21.wav" \
     http://127.0.0.1:7860/api/predict-file
```

---

## Reproducing this

```bash
git clone <this repo> && cd voice-emotion-cnn
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate

# CUDA build; drop the index-url for CPU-only
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/get_data.py          # RAVDESS from Zenodo, ~200 MB
python -m src.train --all           # ~6 min total on an RTX 3060
python -m pytest tests -q
python -m uvicorn app.main:app --port 7860
```

The corpus is not committed: it is 200 MB, and the RAVDESS licence
(CC BY-NC-SA 4.0) does not permit redistribution.

## Layout

```
src/config.py      every constant training and serving must agree on
src/features.py    waveform -> log-mel; imported by training AND the server
src/data.py        RAVDESS parsing, speaker-disjoint splits, cached dataset
src/models.py      ScratchCNN, TransferResNet
src/train.py       training loop, checkpointing, cross-validation
src/evaluate.py    UAR, confusion matrices, reports
app/main.py        FastAPI inference server
app/static/        recording page, live spectrogram, result rendering
scripts/get_data.py
tests/             23 tests: split disjointness, feature shape, train/serve parity
reports/           metrics and confusion matrices produced by training
```

## Tests

```bash
python -m pytest tests -q     # 23 passed
```

The one worth reading is `test_app_and_training_compute_identical_features`. If
the server's front end ever drifts from the training front end, the deployed
model is fed a distribution it never saw, nothing raises an exception, and the
demo just quietly gets worse. That test makes the drift impossible.

## Deployment

One image serves both targets. It installs the CPU wheels -- the CUDA build is
2.5 GB -- and takes its port from `$PORT`, defaulting to 7860.

**Google Cloud Run** (current deployment):

```bash
gcloud run deploy voice-emotion --source .     --region europe-west9 --allow-unauthenticated     --port 8080 --memory 2Gi --cpu-boost
```

Cloud Run scales to zero, so the first request after an idle period pays a cold
start of roughly twenty seconds while the container boots and loads the three
checkpoints. The page polls `/health` and says it is waking the server rather
than rendering a dead interface -- without that, a first-time visitor
reasonably concludes the demo is broken.

**Hugging Face Spaces** (Docker SDK) builds from the same repository, since the
README carries the frontmatter it needs:

```bash
git remote add space https://huggingface.co/spaces/<user>/voice_emotion
git push space main
```

## Who did what

<!-- TODO: fill in before submitting -->

| | Contribution |
|---|---|
| *name* | |
| *name* | |
| *name* | |

## References

- Livingstone & Russo, *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, PLoS ONE 13(5), 2018. [Zenodo record](https://zenodo.org/records/1188976)
- Park et al., *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*, Interspeech 2019.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Dumoulin & Visin, *A guide to convolution arithmetic for deep learning*, arXiv:1603.07285, 2016.
- Stanford CS231n, [Convolutional Neural Networks notes](https://cs231n.github.io/convolutional-networks/).
