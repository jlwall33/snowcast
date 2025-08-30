#!/usr/bin/env bash
set -euo pipefail
cd ~/snowcast

# 1) generate synthetic NPZ tiles if none exist
mkdir -p data/npz_samples
npz_count=$(ls -1 data/npz_samples/*.npz 2>/dev/null | wc -l || true)
if [ "$npz_count" -eq 0 ]; then
  cat > /tmp/generate_synthetic_npz.py <<'PY'
#!/usr/bin/env python3
import os, numpy as np
os.makedirs("data/npz_samples", exist_ok=True)
def smooth(a, iters=2):
    for _ in range(iters):
        a = np.pad(a, 1, mode='reflect')
        a = (a[0:-2,0:-2] + a[0:-2,1:-1] + a[0:-2,2:] +
             a[1:-1,0:-2] + a[1:-1,1:-1] + a[1:-1,2:] +
             a[2:,0:-2]   + a[2:,1:-1]   + a[2:,2:]) / 9.0
    return a
N = 40; C = 6; H = 128; W = 128
for i in range(N):
    base = np.random.randn(H, W).astype(np.float32)
    base = smooth(base, iters=6)
    chs = [base]
    chs.append(np.clip(base * 0.5 + np.random.randn(H,W)*0.05, -3, 3))
    chs.append(smooth(np.random.randn(H,W).astype(np.float32), iters=4))
    gx, gy = np.gradient(base)
    chs.append(gx.astype(np.float32)); chs.append(gy.astype(np.float32))
    xx = np.linspace(-1,1,W); yy = np.linspace(-1,1,H)[:,None]
    elev = (1 - (xx**2 + yy**2)) * 0.2
    chs.append(elev.astype(np.float32))
    X = np.stack(chs, axis=0)[:C]
    y = smooth(base, iters=4) * 3.0 + elev * 2.0
    y = np.clip(y, 0.0, None).astype(np.float32)
    np.savez_compressed(f"data/npz_samples/sample_{i:03d}.npz", X=X, y=y)
print("Wrote synthetic samples: ", len(list(__import__('glob').glob('data/npz_samples/*.npz'))))
PY
  python -u /tmp/generate_synthetic_npz.py
fi

# 2) small training script (uses torch CPU if no GPU)
cat > ~/snowcast/train_smoke.py <<'PY'
#!/usr/bin/env python3
import glob, numpy as np, os
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NPZDataset(Dataset):
    def __init__(self, pattern="data/npz_samples/*.npz"):
        self.files = sorted(glob.glob(pattern))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        X = d['X'].astype('float32')
        y = d['y'].astype('float32')
        return torch.from_numpy(X), torch.from_numpy(y)[None,...]

class SmallNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x): return self.net(x)

def train_one_epoch(model, dl, optim, lossfn, device):
    model.train(); total=0.0
    for xb,yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb); loss = lossfn(pred, yb)
        optim.zero_grad(); loss.backward(); optim.step()
        total += float(loss.item()) * xb.size(0)
    return total / len(dl.dataset)

def main():
    ds = NPZDataset(); dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    sample_X, _ = ds[0]; in_ch = sample_X.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNet(in_ch=in_ch).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.L1Loss()
    print("Device:", device, "samples:", len(ds), "in_ch:", in_ch)
    loss = train_one_epoch(model, dl, optim, lossfn, device)
    print("epoch0 loss:", loss)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/smoke.pth")
    print("Saved models/smoke.pth")
if __name__=='__main__': main()
PY

# ensure torch is present (prefer binary wheels)
python -m pip install --no-cache-dir --prefer-binary torch torchvision || true

# run training (short)
python -u ~/snowcast/train_smoke.py

# 3) inference save png
cat > ~/snowcast/infer_smoke.py <<'PY'
import numpy as np, torch
from matplotlib import pyplot as plt
from train_smoke import SmallNet, NPZDataset
ds = NPZDataset(); X,y = ds[0]
in_ch = X.shape[0]
model = SmallNet(in_ch=in_ch)
model.load_state_dict(torch.load("models/smoke.pth", map_location="cpu"))
model.eval()
with torch.no_grad():
    xb = torch.from_numpy(np.expand_dims(X,0))
    pred = model(xb).squeeze().numpy()
plt.imshow(pred, cmap='viridis'); plt.colorbar()
plt.title("Smoke pred")
plt.savefig("out_smoke_pred.png")
print("Wrote out_smoke_pred.png")
PY

python -u ~/snowcast/infer_smoke.py
ls -lh out_smoke_pred.png || true

echo "=== DONE: synthetic pipeline complete (check out_smoke_pred.png) ==="
