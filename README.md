# ECG Atrial Fibrillation Detection (1D ResNet)

Classifies a 30 s single-lead ECG as **Normal, AF, Other rhythm, or Noisy**.
Built with PyTorch and a Streamlit demo. Educational project, not a medical device.

## Results
- CinC 2017 held-out test split: challenge F1 0.820, accuracy 0.847, AF-vs-rest ROC-AUC 0.974
- External check on MIT-BIH AF (30 s windows, AF vs not-AF): sensitivity 0.844, specificity 0.996
- Limits: Noisy class is weakest (F1 about 0.57); some wide-QRS recordings are still missed

## How it works
Preprocessing: 0.5-40 Hz bandpass, resample to 300 Hz, 30 s window, z-score.
Model: 1D ResNet (about 776k parameters) trained with shift, scale, noise and random
polarity-flip augmentation, class-weighted loss and a One-Cycle learning-rate schedule.

## Run the demo
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

## Data
PhysioNet/CinC Challenge 2017 and the MIT-BIH Atrial Fibrillation Database (PhysioNet).
The 8 files in `samples/` are small examples from the CinC 2017 data, included for the demo.