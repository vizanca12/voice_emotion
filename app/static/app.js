/* Voice emotion recognition -- recording page.
 *
 * Two separate audio paths run here, on purpose:
 *   1. an AnalyserNode drives the live scrolling spectrogram while you speak;
 *   2. a MediaRecorder captures the clip, which is decoded, resampled to
 *      16 kHz in the browser and posted as raw float32.
 * Resampling client-side means the server never has to decode WebM/Opus, so
 * the container needs no ffmpeg.
 */

const SAMPLE_RATE = 16000;
const MAX_SECONDS = 8;
const F_MAX_DISPLAY = 8000;

const el = (id) => document.getElementById(id);
const recordBtn = el('recordBtn');
const uploadBtn = el('uploadBtn');
const fileInput = el('fileInput');
const timerEl = el('timer');
const hintEl = el('hint');
const errorEl = el('error');
const liveStatus = el('liveStatus');
const resultsEl = el('results');

let audioCtx = null;
let mediaRecorder = null;
let analyser = null;
let stream = null;
let chunks = [];
let rafId = null;
let startedAt = 0;
let timerId = null;

/* ---------------------------------------------------------------- colormap */

// Magma, subsampled to 16 stops and interpolated. Perceptually uniform, and
// it reads well on a dark page -- quiet regions stay genuinely dark.
const MAGMA = [
  [0, 0, 4], [12, 8, 38], [35, 12, 74], [64, 10, 103], [93, 18, 110],
  [122, 28, 109], [151, 37, 103], [180, 47, 91], [206, 61, 75], [227, 82, 59],
  [241, 109, 47], [250, 140, 45], [253, 172, 62], [254, 203, 92], [251, 232, 138],
  [252, 253, 191],
];

function magma(t) {
  const x = Math.max(0, Math.min(1, t)) * (MAGMA.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = MAGMA[i];
  const b = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

/* ---------------------------------------------------- live spectrogram */

function startLiveSpectrogram() {
  const canvas = el('live');
  const ctx = canvas.getContext('2d', { willReadFrequently: false });
  const W = canvas.width, H = canvas.height;
  const COL = 3;

  ctx.fillStyle = '#05060c';
  ctx.fillRect(0, 0, W, H);

  const bins = new Uint8Array(analyser.frequencyBinCount);
  const binHz = audioCtx.sampleRate / 2 / analyser.frequencyBinCount;
  const melMin = hzToMel(20);
  const melMax = hzToMel(Math.min(F_MAX_DISPLAY, audioCtx.sampleRate / 2));

  // Precompute the row -> FFT-bin map once; it never changes during a take.
  const rowBin = new Int32Array(H);
  for (let y = 0; y < H; y++) {
    const frac = 1 - y / (H - 1);                       // bottom row = lowest frequency
    const hz = melToHz(melMin + frac * (melMax - melMin));
    rowBin[y] = Math.min(analyser.frequencyBinCount - 1, Math.round(hz / binHz));
  }

  const column = ctx.createImageData(COL, H);

  function draw() {
    analyser.getByteFrequencyData(bins);
    ctx.drawImage(canvas, -COL, 0);                     // scroll left

    for (let y = 0; y < H; y++) {
      const v = bins[rowBin[y]] / 255;
      const [r, g, b] = magma(v);
      for (let x = 0; x < COL; x++) {
        const o = (y * COL + x) * 4;
        column.data[o] = r; column.data[o + 1] = g; column.data[o + 2] = b; column.data[o + 3] = 255;
      }
    }
    ctx.putImageData(column, W - COL, 0);
    rafId = requestAnimationFrame(draw);
  }
  draw();
}

/* ------------------------------------------------------------- recording */

async function startRecording() {
  hideError();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false },
    });
  } catch (e) {
    showError('Microphone unavailable: ' + e.message + ' — you can still upload a file.');
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.55;
  source.connect(analyser);

  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();

  startedAt = performance.now();
  startLiveSpectrogram();
  timerId = setInterval(tick, 100);

  recordBtn.textContent = 'Stop';
  recordBtn.classList.add('recording');
  liveStatus.textContent = 'recording';
  hintEl.textContent = 'Listening… press Stop when you are done.';
}

function tick() {
  const s = (performance.now() - startedAt) / 1000;
  timerEl.textContent = s.toFixed(1) + 's';
  if (s >= MAX_SECONDS) stopRecording();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
  clearInterval(timerId);
  cancelAnimationFrame(rafId);
  recordBtn.textContent = 'Start recording';
  recordBtn.classList.remove('recording');
  liveStatus.textContent = 'analysing…';
}

async function handleStop() {
  stream.getTracks().forEach((t) => t.stop());
  const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
  try {
    await analyse(await blob.arrayBuffer());
  } catch (e) {
    showError('Could not analyse the recording: ' + e.message);
    liveStatus.textContent = 'idle';
  }
}

/* -------------------------------------------------- decode, resample, post */

async function toMono16k(arrayBuffer) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  ctx.close();

  const frames = Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE));
  const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE);
  const src = offline.createBufferSource();
  src.buffer = decoded;                 // mixes down to the context's 1 channel
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

async function analyse(arrayBuffer) {
  recordBtn.disabled = true;
  const pcm = await toMono16k(arrayBuffer);

  const res = await fetch(`/api/predict?sr=${SAMPLE_RATE}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: pcm.buffer,
  });
  recordBtn.disabled = false;

  if (!res.ok) {
    showError('Server said: ' + (await res.text()));
    liveStatus.textContent = 'idle';
    return;
  }

  render(await res.json());
  liveStatus.textContent = 'done';
  hintEl.textContent = 'Record again to compare, or try a different emotion.';
}

/* ----------------------------------------------------------------- render */

function drawMel(spec) {
  const canvas = el('melCanvas');
  const { n_mels: H, n_frames: W } = spec;
  canvas.width = W;
  canvas.height = H;

  const raw = atob(spec.data);
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);

  for (let m = 0; m < H; m++) {
    for (let t = 0; t < W; t++) {
      const v = raw.charCodeAt(m * W + t) / 255;
      const [r, g, b] = magma(v);
      // Row 0 is the lowest mel band, but low frequencies belong at the bottom.
      const o = ((H - 1 - m) * W + t) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawWave(envelope) {
  const canvas = el('waveCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#05060c';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#4ecdc4';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < envelope.length; i++) {
    const x = (i / (envelope.length - 1)) * W;
    const h = envelope[i] * (H / 2 - 2);
    ctx.moveTo(x, H / 2 - h);
    ctx.lineTo(x, H / 2 + h);
  }
  ctx.stroke();
}

function renderModels(data) {
  const container = el('models');
  container.innerHTML = '';

  for (const [name, pred] of Object.entries(data.predictions)) {
    const entries = Object.entries(pred.probs).sort((a, b) => b[1] - a[1]);
    const card = document.createElement('div');
    card.className = 'model-card';
    card.innerHTML = `
      <div class="name">${name}</div>
      <div class="desc">${(META.models.find((m) => m.name === name) || {}).description || ''}</div>
      <div class="verdict">${pred.top}
        <small>${(pred.confidence * 100).toFixed(0)}% confident · runner-up: ${pred.runner_up}</small>
      </div>
      <div class="bars">
        ${entries.map(([emo, p], i) => `
          <div class="bar-row ${i === 0 ? 'top' : ''}">
            <span class="label">${emo}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
            <span class="pct">${(p * 100).toFixed(0)}%</span>
          </div>`).join('')}
      </div>`;
    container.appendChild(card);
  }
}

function render(data) {
  drawMel(data.spectrogram);
  drawWave(data.envelope);
  el('durLabel').textContent = `you recorded ${data.duration.toFixed(1)} s`;
  renderModels(data);
  resultsEl.classList.remove('hidden');
}

/* ------------------------------------------------------------------- meta */

let META = { models: [] };

async function loadMeta() {
  try {
    META = await (await fetch('/api/meta')).json();
  } catch {
    return;
  }
  const body = el('metricsBody');
  if (!META.models.length) {
    body.innerHTML = '<tr><td colspan="4" style="color:var(--muted)">' +
      'No checkpoint loaded — run <code>python -m src.train --all</code>.</td></tr>';
    return;
  }
  const pct = (v) => (v == null ? '—' : (v * 100).toFixed(1) + ' %');
  body.innerHTML = META.models.map((m) => `
    <tr>
      <td>${m.name}</td>
      <td style="color:var(--muted)">${m.description}</td>
      <td class="num">${pct(m.uar)}</td>
      <td class="num">${pct(m.accuracy)}</td>
    </tr>`).join('');
}

/* ------------------------------------------------------------------ wiring */

function showError(msg) { errorEl.textContent = msg; errorEl.classList.remove('hidden'); }
function hideError() { errorEl.classList.add('hidden'); }

recordBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
  else startRecording();
});

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  hideError();
  liveStatus.textContent = 'analysing…';
  try {
    await analyse(await file.arrayBuffer());
  } catch (err) {
    showError('Could not read that file: ' + err.message);
  }
});

loadMeta();
