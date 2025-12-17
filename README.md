# Keyword Spotting (KWS) — “yes” / “no” (CNN + Log-Mel Spectrograms)

This project trains a small **Convolutional Neural Network (CNN)** to recognize spoken keywords from 1-second audio clips and then runs **live microphone inference** to detect commands in real time.

Out of the box, it detects two words:
- **yes**
- **no**

The script will automatically:
1. **Download** a small speech dataset (TensorFlow “mini_speech_commands”).
2. **Train** a CNN on Log-Mel spectrograms.
3. **Save** the trained model + normalization parameters.
4. **Start listening** on your microphone and print detections when confidence is high.

---

## Quick start

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `PyAudio` can be the trickiest dependency to install (it depends on PortAudio).  

### 3) Run
```bash
python keyword_spotter.py
```

- If no model files are found, the script trains a new model.
- If model files exist, it loads them and immediately starts live listening.

Stop live mode with **Ctrl+C**.

---

## What the script produces

After training you should see two files in the project directory:

- `kws_cnn_model.h5` — trained CNN model
- `kws_normalization_params.npz` — normalization parameters (`mean`, `std`)

These are reused for live inference so you don’t retrain every time.

---

## How it works (pipeline)

### 1) Dataset
The script downloads and extracts TensorFlow’s **mini_speech_commands** dataset automatically, then filters only the folders for the target classes (`yes`, `no`).

Dataset path used by the code:
```
data/mini_speech_commands_extracted/mini_speech_commands/
```

### 2) Audio preprocessing → Log-Mel spectrogram “images”
For each `.wav` file:
1. Load and resample to **16 kHz**
2. Force fixed length **1.0 s** (pad with zeros or trim)
3. Convert to **Mel spectrogram**
4. Convert power → **dB (log scale)**
5. Add a channel dimension so the CNN can treat it like an image:
   `(mel_bins, time_frames, 1)`

### 3) Normalization (Z-score)
The training pipeline computes per-feature normalization:
- `mean = mean(X, axis=0)`
- `std = std(X, axis=0)`
- `X_norm = (X - mean) / std`

The same `mean/std` are used to normalize live microphone spectrograms.

### 4) CNN model
The network is a compact CNN classifier:
- Conv2D → MaxPool → Dropout
- Conv2D → MaxPool → Dropout
- Conv2D → MaxPool → Dropout
- Flatten → Dense(256) → Dropout → Dense(num_classes, softmax)

Loss: categorical cross-entropy  
Optimizer: Adam  
Metric: accuracy

### 5) Live microphone inference
Live mode captures **exactly 1 second** of audio at a time (16,000 samples at 16 kHz), runs the same preprocessing, then predicts the class probabilities.

It prints a detection only when:
- model confidence `> CONFIDENCE_THRESHOLD`
- and at least `DEBOUNCE_TIME` seconds passed since the last detection (prevents spam)

---

## Configuration (edit at the top of `keyword_spotter.py`)

### Keywords (classes)
```python
TARGET_WORDS = ['yes', 'no']
```

### Audio settings
```python
SAMPLE_RATE = 16000
AUDIO_LENGTH_SECONDS = 1.0
```

### Spectrogram settings
```python
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
```

### Live detection behavior
```python
CONFIDENCE_THRESHOLD = 0.85  # higher = fewer false positives, but may miss words
DEBOUNCE_TIME = 2.0          # minimum seconds between detections
```

---

## Customizing the project (use your own keywords)

You can change the recognized commands by editing:
```python
TARGET_WORDS = ['word1', 'word2', ...]
```

**Important:** the dataset folder must contain matching subfolders:
```
.../mini_speech_commands/word1/*.wav
.../mini_speech_commands/word2/*.wav
...
```

After changing `TARGET_WORDS`, delete old artifacts so you don’t load a model trained for different labels:
```bash
rm -f kws_cnn_model.h5 kws_normalization_params.npz
```

Then run the script again to retrain:
```bash
python keyword_spotter.py
```

---

## Notes / known caveats

### 1) Chunking strategy (live mode)
Live detection runs on **non-overlapping 1-second windows**.  
This is simple and works well for demos, but a spoken word can sometimes fall across the boundary between chunks. A common improvement is using a rolling buffer and overlapping windows.

### 2) Label mapping consistency
For reliable class-name output, keep a consistent mapping between:
- `label_to_id` (e.g. `"yes" -> 0`)
- `id_to_label` (e.g. `0 -> "yes"`)

The model outputs class indices (e.g. `0` or `1`), so the project must translate them back into human-readable labels.  
If the mapping used during training differs from the mapping used during live inference, the script may print the wrong word even when the prediction index is correct.

To avoid this, keep the mapping deterministic (based on `TARGET_WORDS`) or save/load it alongside the model (e.g. as a JSON file).