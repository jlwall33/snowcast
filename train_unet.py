#!/usr/bin/env python3
import glob, os, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.unet import UNet

class NPZDataset(Dataset):
    def __init__(self, pattern="data/npz_samples/*.npz"):
        self.files = sorted(glob.glob(pattern))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        X = d['X'].astype('float32')
        y = d['y'].astype('float32')
        return torch.from_numpy(X), torch.from_numpy(y)[None,...]

def train_one_epoch(model, dl, optim, lossfn, device):
    model.train()
    total = 0.0
    for xb,yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = lossfn(pred, yb)
        optim.zero_grad(); loss.backward(); optim.step()
        total += float(loss.item()) * xb.size(0)
    return total / len(dl.dataset)

def main():
    ds = NPZDataset()
    if len(ds)==0:
        raise SystemExit("No NPZ samples found in data/npz_samples")
    sample_X,_ = ds[0]
    in_ch = sample_X.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=in_ch).to(device)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.L1Loss()
    print("Device:", device, "samples:", len(ds), "in_ch:", in_ch)
    loss = train_one_epoch(model, dl, optim, lossfn, device)
    print("epoch0 loss:", loss)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_smoke.pth")
    print("Saved models/unet_smoke.pth")

if __name__=="__main__":
    main()
