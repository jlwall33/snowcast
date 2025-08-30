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
