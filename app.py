import io
import json

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from ecg_model import load_model, CLASS_NAMES
from ecg_infer import predict

st.set_page_config(page_title="ECG AF Detector", layout="centered")


@st.cache_resource
def get_model():
    return load_model('model_v4.pt')


@st.cache_data
def get_samples():
    with open('samples/samples.json') as f:
        return json.load(f)


def read_upload(file):
    if file.name.lower().endswith('.npy'):
        arr = np.load(file, allow_pickle=False)
    else:
        arr = np.loadtxt(io.StringIO(file.getvalue().decode('utf-8')), delimiter=',')
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[0] if arr.shape[0] == 1 else arr[:, 0]
    return arr.ravel()


st.title("ECG Atrial Fibrillation Detector")
st.caption("Educational demo: a 1D ResNet trained on the PhysioNet/CinC 2017 challenge data. "
           "Not a medical device and not for diagnosis.")

model = get_model()
samples = get_samples()

source = st.radio("Input", ["Sample recordings", "Upload your own"], horizontal=True)
raw, fs, true_label = None, 300, None

if source == "Sample recordings":
    i = st.selectbox("Recording", range(len(samples)),
                     format_func=lambda k: f"{samples[k]['record']} (labeled {samples[k]['true']})")
    raw = np.load('samples/' + samples[i]['file'])
    fs = samples[i]['fs']
    true_label = samples[i]['true']
else:
    up = st.file_uploader("Single-lead ECG (.npy, .csv or .txt; one value per line or comma-separated)",
                          type=['npy', 'csv', 'txt'])
    fs = st.number_input("Sampling rate (Hz)", min_value=100, max_value=2000, value=300, step=1)
    if up is not None:
        try:
            raw = read_upload(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")

if raw is not None:
    try:
        res = predict(model, raw, fs)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    probs, info = res['probs'], res['info']

    st.subheader(f"Prediction: {res['label']} ({probs.max():.0%} confidence)")
    if true_label is not None:
        st.caption(f"Label in the dataset: {true_label}")
    if info['seconds'] > 30:
        st.info("The recording is longer than 30 s, so only the first 30 s were analysed.")
    if info['tiled']:
        st.warning(f"The recording is only {info['seconds']:.0f} s, so it was repeated to fill 30 s. "
                   "Treat the result with extra caution.")

    fig, ax = plt.subplots(figsize=(6, 2.2))
    colors = ['#1f77b4' if k == int(probs.argmax()) else '#c7c7c7' for k in range(4)]
    ax.barh(range(4), probs, color=colors)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    for k, p in enumerate(probs):
        ax.text(min(p + 0.02, 0.85), k, f"{p:.0%}", va='center')
    ax.set_xlabel("Model probability")
    st.pyplot(fig)
    plt.close(fig)

    x = res['input']
    start = st.slider("Waveform window start (s)", 0, 20, 0)
    seg = x[start * 300:(start + 10) * 300]
    t = start + np.arange(len(seg)) / 300
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(t, seg, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("Filtered ECG as seen by the model (10 s window)")
    st.pyplot(fig)
    plt.close(fig)

with st.expander("About this model and its limits"):
    st.markdown(
        """
**Classes:** Normal rhythm, AF (atrial fibrillation), Other (any other abnormal rhythm),
Noisy (too noisy to interpret).

**Measured results**
- CinC 2017 held-out test split: challenge F1 0.82 (mean F1 of Normal, AF and Other),
  AF recall 0.80, AF precision 0.77.
- External check on MIT-BIH AF (23 Holter recordings, 30 s windows): AF sensitivity 0.84,
  specificity 0.996.

**Limits**
- One lead only, and only the first 30 s of a recording is analysed.
- The Noisy class is the weakest (F1 about 0.57).
- Some recordings with wide or unusual QRS shapes are still missed or called Other.
- Results come from research datasets. This is not a diagnostic tool.
        """
    )