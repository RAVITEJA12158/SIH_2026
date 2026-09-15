"""
GeoSync AI - Specialized Model (2nd escalation tier)
------------------------------------------------------
When SegFormer's confidence is low on a region (ambiguous boundaries,
shadow/tree occlusion, dense clustering), the pipeline escalates to a
SMALLER, TARGETED model trained specifically on hard examples rather
than asking the generalist SegFormer to do better. This mirrors the
doc: "If SegFormer does not perform sufficiently for a particular
difficult feature class, route that feature to a specialized model
designed for that specific extraction challenge."

Architecture: compact U-Net (fewer params than SegFormer, faster to
retrain/adapt per feature-class) with dilated convolutions to better
see through small occluding blobs.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TILE = 64


def _conv_block(cin, cout, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class MiniUNet(nn.Module):
    """Small dilated U-Net for hard/occluded footprint regions."""

    def __init__(self, in_ch=3, base=24):
        super().__init__()
        self.enc1 = nn.Sequential(_conv_block(in_ch, base), _conv_block(base, base))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(_conv_block(base, base * 2, dilation=2), _conv_block(base * 2, base * 2, dilation=2))
        self.pool2 = nn.MaxPool2d(2)
        self.bott = nn.Sequential(_conv_block(base * 2, base * 4, dilation=4), _conv_block(base * 4, base * 4, dilation=4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(_conv_block(base * 4, base * 2), _conv_block(base * 2, base * 2))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(_conv_block(base * 2, base), _conv_block(base, base))
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bott(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class HardTileDataset(Dataset):
    """Synthesizes extra-difficult tiles: heavier occlusion/shadow density
    than the main scene, so the specialized model actually learns
    something the generalist SegFormer wasn't optimized for."""

    def __init__(self, n_samples=260, tile=TILE, train=True):
        from synthetic_data import generate_building_layout, synthesize_orthomosaic
        self.samples = []
        rng = np.random.default_rng(7 if train else 8)
        n_scenes = 6
        for s in range(n_scenes):
            polys = generate_building_layout(n=14, size=256)
            img, mask = synthesize_orthomosaic(polys, size=256, noise_level=22, add_occlusion=True)
            # amplify occlusion further for "hard" training signal
            for _ in range(10):
                tx, ty = rng.uniform(0, 256, size=2)
                r = rng.uniform(10, 26)
                yy, xx = np.ogrid[:256, :256]
                blob = (xx - tx) ** 2 + (yy - ty) ** 2 <= r ** 2
                img[blob] = (img[blob] * 0.25 + np.array([20, 60, 20]) * 0.75).astype(np.uint8)
            for _ in range(n_samples // n_scenes):
                y = rng.integers(0, 256 - tile)
                x = rng.integers(0, 256 - tile)
                self.samples.append((img[y:y + tile, x:x + tile].copy(), mask[y:y + tile, x:x + tile].copy()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, m = self.samples[idx]
        img = img.astype(np.float32) / 255.0
        if np.random.rand() > 0.5:
            img, m = img[:, ::-1].copy(), m[:, ::-1].copy()
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        return img_t, torch.from_numpy(m.astype(np.int64))


def train_specialized_model(epochs=8, device="cpu", verbose=True):
    model = MiniUNet().to(device)
    ds = HardTileDataset(train=True)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        if verbose:
            print(f"  [Specialized] epoch {ep+1}/{epochs}  loss={tot/len(ds):.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict_patch(model, patch_uint8, device="cpu"):
    """patch_uint8: (h,w,3) uint8, any size (model is fully convolutional
    at inference since no pooling-size mismatch for even dims)."""
    h, w = patch_uint8.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4  # pad so two 2x pools divide evenly
    padded = np.pad(patch_uint8, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    t = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    logits = model(t)
    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    prob = prob[:h, :w]
    return prob


if __name__ == "__main__":
    m = train_specialized_model(epochs=8)
    torch.save(m.state_dict(), "/home/claude/geosync_ai/models/specialized_model.pt")
    print("Specialized model trained and saved.")
