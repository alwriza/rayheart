import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tsai.models.XResNet1d import xresnet1d101

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "processed data"
MODEL = BASE_DIR / "trained model"
MODEL.mkdir(exist_ok=True)
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH = 128
LR = 1e-3
PATIENCE = 8


def load_split(name):
    X = np.load(DATA / f"X_{name}.npy")            # (N, 1000, 12) from make_dataset.py
    Y = np.load(DATA / f"Y_{name}.npy")            # (N, 5)
    X = np.transpose(X, (0, 2, 1))          # -> (N, 12, 1000): conv wants (batch, channels, length)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def loader(name, shuffle):
    X, Y = load_split(name)
    return DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=shuffle)


def macro_auc(y_true, y_prob):
   
    aucs = []
    for i, c in enumerate(SUPER):
        if y_true[:, i].sum() > 0:
            aucs.append((c, roc_auc_score(y_true[:, i], y_prob[:, i])))
    macro = float(np.mean([a for _, a in aucs]))
    return macro, aucs


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    P, T = [], []
    for x, y in dl:
        P.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
        T.append(y.numpy())
    return macro_auc(np.concatenate(T), np.concatenate(P))


def main():
    tr = loader("train", shuffle=True)
    va = loader("val", shuffle=False)
    te = loader("test", shuffle=False)

    model = xresnet1d101(c_in=12, c_out=5).to(DEVICE)   
    criterion = nn.BCEWithLogitsLoss()                  
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best, bad_epochs = 0.0, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in tr:
            optimizer.zero_grad()
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()

        v_macro, v_aucs = evaluate(model, va)
        scheduler.step(v_macro)
        per_class = "  ".join(f"{c}:{a:.3f}" for c, a in v_aucs)
        print(f"epoch {ep:02d}  val macro-AUC {v_macro:.4f}  |  {per_class}")

        if v_macro > best:
            best, bad_epochs = v_macro, 0
            torch.save(model.state_dict(), MODEL / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stop at epoch {ep} (best val macro-AUC {best:.4f})")
                break

    
    model.load_state_dict(torch.load(MODEL / "best.pt", map_location=DEVICE))
    t_macro, t_aucs = evaluate(model, te)
    print(f"\nTEST macro-AUC: {t_macro:.4f}")
    for c, a in t_aucs:
        print(f"  {c:5s} {a:.4f}")


if __name__ == "__main__":
    main()