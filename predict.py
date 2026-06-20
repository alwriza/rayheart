import numpy as np
import torch
from pathlib import Path
from tsai.models.XResNet1d import xresnet1d101

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "processed_data"        
MODEL = BASE_DIR / "trained_model"

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = xresnet1d101(c_in=12, c_out=5).to(DEVICE)
model.load_state_dict(torch.load(MODEL / "best.pt", map_location=DEVICE))
model.eval()

X = np.load(DATA / "X_test.npy")                             
x = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)   

with torch.no_grad():
    probs = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()   

for i in range(min(5, len(probs))):
    pred = {c: round(float(p), 3) for c, p in zip(SUPER, probs[i])}
    print(f"sample {i}: {pred}")