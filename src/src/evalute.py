import argparse, torch
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.dataset import ImageFolderAlb
from src.transforms import val_transforms
from src.metrics import compute_metrics
from src.common import save_json
import numpy as np

def main(args):
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg.weights, map_location=device)
    class_to_idx = ckpt['class_to_idx']
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    from src.build import build_model
    model = build_model("efficientnet_b0", num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt['model']); model.to(device); model.eval()

    t_val = val_transforms(cfg.img_size, [0.485,0.456,0.406], [0.229,0.224,0.225])
    ds = ImageFolderAlb(Path(cfg.data_root)/"test", t_val, class_to_idx=class_to_idx)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y)

    metrics = compute_metrics(y_true, y_pred, labels=list(range(len(class_to_idx))))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    save_json(metrics, f"{cfg.output_dir}/metrics.json")
    print("Top-1 Acc:", metrics["top1_acc"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    main(args)
