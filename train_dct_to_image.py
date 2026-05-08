import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Configuration
# ----------------------------
IMAGE_DIR = Path("grayscale_100x100_images")
DCT_DIR = Path("dct_results_100x100")

TEST_DCT_PATH = Path("dct_test.npy")
OUTPUT_IMAGE_PATH = Path("predicted_image_test.jpg")

NUM_SAMPLES = 2000
IMAGE_SIZE = 100
BATCH_SIZE = 16
NUM_EPOCHS = 90
LEARNING_RATE = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Dataset
# ----------------------------
class DCTImageDataset(Dataset):
    def __init__(self, image_dir, dct_dir, num_samples):
        self.image_dir = image_dir
        self.dct_dir = dct_dir
        self.num_samples = num_samples

        # Estimate normalization statistics for DCT coefficients
        all_dct = []
        for i in range(1, num_samples + 1):
            dct = np.load(dct_dir / f"dct_{i:05d}.npy")
            all_dct.append(dct)

        all_dct = np.stack(all_dct)
        self.dct_mean = all_dct.mean()
        self.dct_std = all_dct.std() + 1e-8

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_id = idx + 1

        img_path = self.image_dir / f"image_{sample_id:05d}.jpg"
        dct_path = self.dct_dir / f"dct_{sample_id:05d}.npy"

        img = Image.open(img_path).convert("L")
        img = np.asarray(img, dtype=np.float32) / 255.0

        dct = np.load(dct_path).astype(np.float32)
        dct = (dct - self.dct_mean) / self.dct_std

        img = torch.from_numpy(img).unsqueeze(0)
        dct = torch.from_numpy(dct).unsqueeze(0)

        return dct, img


# ----------------------------
# Model
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        z_norm = self.norm(z)
        z_attn, _ = self.attn(z_norm, z_norm, z_norm)
        z = z + z_attn
        return z.transpose(1, 2).reshape(B, C, H, W)


class UNetWithAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(128, 256),
            AttentionBlock(256, num_heads=4),
            ConvBlock(256, 256),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)              # 100x100
        e2 = self.enc2(self.pool(e1))  # 50x50
        e3 = self.enc3(self.pool(e2))  # 25x25

        b = self.bottleneck(self.pool(e3))  # 12x12 for 100x100 input

        u3 = self.up3(b)
        u3 = torch.nn.functional.interpolate(u3, size=e3.shape[-2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = torch.nn.functional.interpolate(u2, size=e2.shape[-2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = torch.nn.functional.interpolate(u1, size=e1.shape[-2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


# ----------------------------
# Training
# ----------------------------
dataset = DCTImageDataset(IMAGE_DIR, DCT_DIR, NUM_SAMPLES)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNetWithAttention().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print(f"Training on {DEVICE}")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for dct, img in loader:
        dct = dct.to(DEVICE)
        img = img.to(DEVICE)

        pred = model(dct)
        loss = criterion(pred, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * dct.size(0)

    avg_loss = total_loss / len(dataset)

    
    print(f"Epoch {epoch:04d} | Loss: {avg_loss:.8f}")


# ----------------------------
# Predict test image
# ----------------------------
model.eval()

test_dct = np.load(TEST_DCT_PATH).astype(np.float32)
test_dct = (test_dct - dataset.dct_mean) / dataset.dct_std

test_dct = torch.from_numpy(test_dct).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(test_dct)

pred_img = pred.squeeze().cpu().numpy()
pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)

Image.fromarray(pred_img, mode="L").save(OUTPUT_IMAGE_PATH)

print(f"Saved predicted image to {OUTPUT_IMAGE_PATH.resolve()}")