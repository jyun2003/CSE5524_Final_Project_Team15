import torch
import torch.nn.functional as F
from model_RFDN_advanced import RFDN as RFDN_advanced
from model_RFDN_baseline import RFDN as RFDN_baseline
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.transforms.functional import resize
import numpy as np
from torch.utils.data import DataLoader
from degradations import downsample_raw, convert_to_tensor, simple_deg_simulation
from blur import apply_psf
from noise import add_natural_noise, add_heteroscedastic_gnoise
from utils import plot_all, load_raw, postprocess_raw, demosaic
from glob import glob
import torchvision.transforms.functional as TF
import os
import argparse

kernels = np.load("kernels.npy", allow_pickle=True)
print(kernels.shape)
# plot_all([k for k in kernels])

OUT_PATH = "results_test/"
# RAWS = sorted(glob("train_raws_10/*.npz"))
RAWS = sorted(glob("test_raws/*.npz"))
MAX_VAL = 2**12 - 1
DOWNSAMPLE = False
print(f"Found {len(RAWS)} test files")


def load_raw(raw_path, max_val=2**12 - 1):
    """
    Loads RAW images saved as '.npz' files and returns the 'raw' array.
    """
    data = np.load(raw_path)
    raw = data["image"]  # adjust the key name based on what's inside your .npz
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
        raw_img = load_raw(self.files[idx])  # shape [H, W, 4]
        assert raw_img.shape[-1] == 4, f"Expected 4 channels, got {raw_img.shape}"

        lr = simple_deg_simulation(raw_img, kernels, down=DOWNSAMPLE)  # expected [H, W, 4]
        lr = torch.from_numpy(lr).permute(2, 0, 1).float()
        hr = torch.from_numpy(raw_img).permute(2, 0, 1).float()

        lr = TF.resize(lr, FIXED_LR_SIZE)
        hr = TF.resize(hr, FIXED_HR_SIZE)

        return lr, hr


def rggb_to_rgb(raw):
    return torch.stack([raw[0], 0.5 * (raw[1] + raw[2]), raw[3]], dim=0)  # R  # G  # B


def to_numpy_image(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def evaluate(model, dataloader, device):
    model.eval()
    psnr_total, ssim_total = 0, 0
    count = 0

    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            for i in range(lr.shape[0]):
                sr_img = rggb_to_rgb(sr[i])
                hr_img = rggb_to_rgb(hr[i])

                sr_np = to_numpy_image(sr_img)
                hr_np = to_numpy_image(hr_img)

                psnr = compare_psnr(hr_np, sr_np, data_range=1.0)
                ssim = compare_ssim(hr_np, sr_np, multichannel=True, data_range=1.0)

                psnr_total += psnr
                ssim_total += ssim
                count += 1

    print(f"Average PSNR: {psnr_total / count:.4f}")
    print(f"Average SSIM: {ssim_total / count:.4f}")


# --------- Run Evaluation ----------

parser = argparse.ArgumentParser(description="Evaluate models on RAW dataset")
parser.add_argument('--model', type=str, required=True, help='load the baseline/advanced model')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model== 'baseline':
    model = RFDN_baseline(in_nc=4, out_nc=4, upscale=4).to(device)
else:
    model = RFDN_advanced(in_nc=4, out_nc=4, upscale=4).to(device)

model.load_state_dict(torch.load("results/rfdn_model_"+args.model+".pth", map_location=device))
print("Model loaded.")

dataset = RAWDataset("test_raws")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

evaluate(model, dataloader, device)
