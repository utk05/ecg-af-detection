import json
import numpy as np
from scipy.signal import resample_poly

from ecg_model import load_model
from ecg_infer import predict

model = load_model('model_v4.pt')
meta = json.load(open('samples/samples.json'))

ok = 0
print("file         true    pred    conf  | flipped | 250 Hz")
for m in meta:
    raw = np.load('samples/' + m['file'])
    r = predict(model, raw, m['fs'])
    rf = predict(model, -raw, m['fs'])
    r250 = predict(model, resample_poly(raw, 5, 6), 250)
    ok += r['label'] == m['true']
    left = f"{m['file']:12s} {m['true']:7s} {r['label']:7s} {r['probs'].max():5.2f}"
    print(left + f" | {rf['label']:7s} | {r250['label']:7s}")
print(f"\n{ok}/{len(meta)} correct")