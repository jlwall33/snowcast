#!/usr/bin/env python3
import glob, os, numpy as np
import torch
from models.unet import UNet
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model
m = UNet(in_channels=6).to(device)
m.load_state_dict(torch.load("models/unet_smoke.pth", map_location=device))
m.eval()

os.makedirs("out_unet", exist_ok=True)
for fn in sorted(glob.glob("data/npz_samples/*.npz"))[:20]:
    d = np.load(fn)
    X = d['X'].astype("float32")
    x = torch.from_numpy(X[None,:,:,:]).to(device)  # shape 1,C,H,W
    with torch.no_grad():
        pred = m(x).cpu().numpy()[0,0,:,:]
    # simple normalization for display
    pmin, pmax = pred.min(), pred.max()
    if pmax > pmin:
        p = (pred - pmin) / (pmax - pmin)
    else:
        p = pred - pmin
    outname = os.path.join("out_unet", os.path.basename(fn).replace(".npz", "_pred.png"))
    plt.imsave(outname, p, cmap='viridis')
    print("wrote", outname)
print("Done.")
