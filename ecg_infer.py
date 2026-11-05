import numpy as np
import torch
from fractions import Fraction
from scipy.signal import butter, sosfiltfilt, resample_poly

from ecg_model import CLASS_NAMES

TARGET_FS = 300
WIN = 30 * TARGET_FS          # 9000 samples = 30 s
MIN_SECONDS = 9               # shortest recording seen in training
BAND = (0.5, 40.0)


def preprocess(signal, fs):
    """Raw single-lead ECG (1D, any amplitude units) sampled at fs Hz -> (9000,) float32 model input."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite values.")
    if fs < 100:
        raise ValueError("Sampling rate must be at least 100 Hz.")
    if len(x) / fs < MIN_SECONDS:
        raise ValueError(f"Recording too short: need at least {MIN_SECONDS} seconds.")
    if np.std(x) < 1e-8:
        raise ValueError("Signal is flat (no variation).")

    seconds = len(x) / fs
    sos = butter(4, BAND, btype='bandpass', fs=fs, output='sos')
    x = sosfiltfilt(sos, x)                                    # same 0.5-40 Hz band as training

    if int(round(fs)) != TARGET_FS:                            # resample to 300 Hz if needed
        r = Fraction(TARGET_FS, int(round(fs))).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator)

    tiled = len(x) < WIN
    x = np.pad(x, (0, WIN - len(x)), mode='wrap') if tiled else x[:WIN]   # first 30 s, or tile if shorter
    x = (x - x.mean()) / (x.std() + 1e-8)                      # z-score
    return x.astype(np.float32), {'seconds': seconds, 'tiled': tiled}


def predict(model, signal, fs):
    x, info = preprocess(signal, fs)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None, None, :])
    probs = torch.softmax(logits, dim=1)[0].numpy()
    return {'probs': probs, 'label': CLASS_NAMES[int(probs.argmax())], 'info': info, 'input': x}