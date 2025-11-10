# src/data/dataset.py (fix imread_rgb)
import cv2, torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

VALID_EXT = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    if img.ndim == 2:
        # grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        # use channel axis index 2, not 11
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected channel count {img.shape[2]} in {path}")
    else:
        raise ValueError(f"Unexpected image ndim {img.ndim} in {path}")
    return img

class ImageFolderAlb(Dataset):
    def __init__(self, root, transform=None, class_to_idx: Dict[str,int]=None):
        self.root = Path(root)
        self.transform = transform
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if not classes:
            raise RuntimeError(f"No class folders under {self.root}")
        if class_to_idx is None:
            class_to_idx = {c:i for i,c in enumerate(classes)}
        self.class_to_idx = class_to_idx
        self.samples: List[Tuple[Path,int]] = []
        for c in classes:
            for p in (self.root/c).glob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXT:
                    self.samples.append((p, class_to_idx[c]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root} with extensions {VALID_EXT}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = imread_rgb(p)
        if self.transform is not None:
            out = self.transform(image=img)
            x = out["image"]
        else:
            img = img.astype(np.float32)/255.0
            img = (img - np.array([0.485,0.456,0.406], dtype=np.float32)) / np.array([0.229,0.224,0.225], dtype=np.float32)
            x = torch.from_numpy(np.transpose(img, (2,0,1))).float()
        return x, y
