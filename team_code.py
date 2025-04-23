import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from glob import glob
import sys

from model_RFDN_baseline import RFDN
from model_RFDN_advanced import RFDN

from degradations import downsample_raw, convert_to_tensor, simple_deg_simulation
from blur import apply_psf
from noise import add_natural_noise, add_heteroscedastic_gnoise
from utils import plot_all, load_raw, postprocess_raw, demosaic

kernels = np.load("kernels.npy", allow_pickle=True)
print(kernels.shape)
# plot_all([k for k in kernels])

OUT_PATH = "results/"
# RAWS = sorted(glob("train_raws_10/*.npz"))
RAWS = sorted(glob("train_raws/*.npz"))
MAX_VAL = 2**12 - 1
DOWNSAMPLE = False
print(f"Found {len(RAWS)} RAW files")


def load_raw(raw_path, max_val=2**12 - 1):
    """
    Loads RAW images saved as '.npz' files and returns the 'raw' array.
    """
    data = np.load(raw_path)
    raw = data["raw"]
    raw = raw / max_val
    raw = np.clip(raw, 0.0, 1.0)
    return raw.astype(np.float32)


FIXED_LR_SIZE = (720, 720)
FIXED_HR_SIZE = (2880, 2880)


class RAWDataset(Dataset):
    def __init__(self, raw_dir):
        self.files = sorted(glob(os.path.join(raw_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        raw_img = load_raw(self.files[idx])
        lr = simple_deg_simulation(raw_img, kernels, down=DOWNSAMPLE)
        lr = torch.from_numpy(lr).float()
        lr = lr.permute(2, 0, 1)
        hr = raw_img
        hr = torch.from_numpy(hr).float()
        hr = hr.permute(2, 0, 1)

        lr = TF.resize(lr, FIXED_LR_SIZE)
        hr = TF.resize(hr, FIXED_HR_SIZE)

        return lr, hr


def visualize_sample(lr, sr, hr, title_prefix="", save_path=None):
    """
    Plots a triplet of LR, SR, and HR 4-channel (RGGB) images side-by-side.
    Saves the figure if save_path is provided.
    """

    def rggb_to_rgb(raw):
        return torch.stack([raw[0], 0.5 * (raw[1] + raw[2]), raw[3]], dim=0)  # R  # G = avg of G1 and G2  # B

    lr_rgb = rggb_to_rgb(lr).permute(1, 2, 0).cpu().numpy()
    sr_rgb = rggb_to_rgb(sr).permute(1, 2, 0).detach().cpu().numpy()
    hr_rgb = rggb_to_rgb(hr).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 4))
    for i, (img, label) in enumerate(zip([lr_rgb, sr_rgb, hr_rgb], ["LR", "SR", "HR"])):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"{title_prefix} {label}")
        plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved image to {save_path}")
    else:
        plt.show()

    plt.close()


def train(model, dataloader, optimizer, loss_fn, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            assert sr.shape == hr.shape, f"Shape mismatch: SR {sr.shape}, HR {hr.shape}"
            loss = loss_fn(sr, hr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # if epoch == 0 and batch_idx == 0:
            # visualize_sample(lr[0], sr[0], hr[0], title_prefix="Start")

            if epoch == 0 and batch_idx == 0:
                visualize_sample(lr[0], sr[0], hr[0], title_prefix="Start", save_path=OUT_PATH + "start_sample.png")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # visualize_sample(lr[0], sr[0], hr[0], title_prefix="End")
    # Save last batch's first sample
    visualize_sample(lr[0], sr[0], hr[0], title_prefix="End", save_path=OUT_PATH + "end_sample.png")


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# raw_dir = "train_raws_10"
raw_dir = "train_raws"
batch_size = 2
epochs = 10
lr_rate = 1e-4

# Data and Model
dataset = RAWDataset(raw_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = RFDN(in_nc=4, out_nc=4, upscale=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr_rate)
loss_fn = torch.nn.L1Loss()

# Train
train(model, dataloader, optimizer, loss_fn, device, epochs=epochs)
# save model
MODEL_SAVE_PATH = "results/rfdn_model_advanced.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
