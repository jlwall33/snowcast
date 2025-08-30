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
