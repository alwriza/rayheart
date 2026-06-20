import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tsai.models.XResNet1d import xresnet1d101

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "processed_data"
MODEL = BASE_DIR / "trained_model"
MODEL.mkdir(exist_ok=True)
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH = 128
LR = 1e-3
PATIENCE = 8
THRESHOLD = 0.5     


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


@torch.no_grad()
def predict(model, dl):
    """Return (y_true, y_prob) arrays of shape (N, 5)."""
    model.eval()
    P, T = [], []
    for x, y in dl:
        P.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
        T.append(y.numpy())
    return np.concatenate(T), np.concatenate(P)


def full_report(y_true, y_prob, threshold=THRESHOLD):
    """Per-class AUC, sensitivity, specificity at the given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, c in enumerate(SUPER):
        t, p = y_true[:, i], y_pred[:, i]
        tp = int(((t == 1) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        tn = int(((t == 0) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")    # true-positive rate
        spec = tn / (tn + fp) if (tn + fp) else float("nan")    # true-negative rate
        auc = roc_auc_score(t, y_prob[:, i]) if t.sum() > 0 else float("nan")
        rows.append((c, auc, sens, spec))
    return rows


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
    yt, yp = predict(model, te)
    rows = full_report(yt, yp, THRESHOLD)

    print(f"\nTEST metrics (sensitivity/specificity at threshold {THRESHOLD})")
    print(f"{'class':6s} {'AUC':>7s} {'Sens':>7s} {'Spec':>7s}")
    for c, auc, sens, spec in rows:
        print(f"{c:6s} {auc:7.4f} {sens:7.4f} {spec:7.4f}")
    print(f"{'MACRO':6s} {np.nanmean([r[1] for r in rows]):7.4f} "
          f"{np.nanmean([r[2] for r in rows]):7.4f} {np.nanmean([r[3] for r in rows]):7.4f}")


if __name__ == "__main__":
    main()