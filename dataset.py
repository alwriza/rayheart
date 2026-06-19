import ast
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PTBXL_ROOT = BASE_DIR / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
LABELS_CSV = BASE_DIR / "ptbxl_master.csv"   
OUT_DIR = BASE_DIR / "processed data"
OUT_DIR.mkdir(exist_ok=True) 
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLING = 100                            


def load_signals(filenames):
    
    sigs = []
    for i, f in enumerate(filenames):
        signal, _meta = wfdb.rdsamp(str(PTBXL_ROOT / f))
        sigs.append(signal.astype(np.float32))
        if (i + 1) % 2000 == 0:
            print(f"  loaded {i + 1}/{len(filenames)}")
    return np.asarray(sigs, dtype=np.float32)


def main():
    df = pd.read_csv(LABELS_CSV, index_col="ecg_id", low_memory=False)
    fname_col = "filename_lr" if SAMPLING == 100 else "filename_hr"

    splits = {}
    for name in ["train", "val", "test"]:
        sub = df[df.split == name]
        print(f"\n[{name}] {len(sub)} records — loading signals...")
        X = load_signals(sub[fname_col].values)
        Y = sub[SUPER].values.astype(np.float32)
        splits[name] = (X, Y)

    # ---- fit per-lead standardizer on TRAIN only, apply to all ----
    Xtr = splits["train"][0]
    mean = Xtr.reshape(-1, 12).mean(axis=0)          # (12,)
    std = Xtr.reshape(-1, 12).std(axis=0) + 1e-8     # (12,)

    for name, (X, Y) in splits.items():
        Xn = (X - mean) / std
        np.save(OUT_DIR / f"X_{name}.npy", Xn.astype(np.float32))
        np.save(OUT_DIR / f"Y_{name}.npy", Y)
        print(f"saved X_{name}.npy {Xn.shape}  Y_{name}.npy {Y.shape}")

    np.save(OUT_DIR / "lead_mean.npy", mean)
    np.save(OUT_DIR / "lead_std.npy", std)
    print("\nDone")


if __name__ == "__main__":
    main()