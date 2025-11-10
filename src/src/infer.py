import torch, cv2
import numpy as np
from pathlib import Path
from src.dataset import imread_rgb
from src.transforms import val_transforms

class InferenceModel:
    def __init__(self, weights_path, model_name="efficientnet_b0", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location=self.device)
        self.class_to_idx = ckpt["class_to_idx"]
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}

        # Pull preprocessing params from training cfg if available
        cfg = ckpt.get("cfg", {}) or {}
        img_size = int(cfg.get("img_size", 160))
        data_cfg = cfg.get("data", {}) or {}
        mean = data_cfg.get("mean", [0.485,0.456,0.406])
        std  = data_cfg.get("std",  [0.229,0.224,0.225])

        import timm
        # Prefer model name from training cfg if available unless explicitly overridden
        cfg_model = (cfg.get("model_name") if isinstance(cfg, dict) else None) or model_name
        self.model = timm.create_model(cfg_model, pretrained=False, num_classes=len(self.class_to_idx))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

        self.tf = val_transforms(img_size, mean, std)
        
    def predict_image(self, img_rgb, topk=5, tta: str = "fast"):
        """Predict with optional lightweight TTA to stabilize probabilities.
        tta modes:
          - None/"off": single pass
          - "fast": original + horizontal flip
          - "full": original + hflip + vflip
        """
        variants = [img_rgb]
        if tta and tta != "off":
            variants.append(np.ascontiguousarray(img_rgb[:, ::-1, :]))  # hflip
            if tta == "full":
                variants.append(np.ascontiguousarray(img_rgb[::-1, :, :]))  # vflip

        use_cuda = (self.device.type == "cuda")
        prob_sum = None
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            for v in variants:
                x = self.tf(image=v)["image"].unsqueeze(0).to(self.device).to(memory_format=torch.channels_last)
                logits = self.model(x)
                prob = torch.softmax(logits, dim=1)
                prob_sum = prob if prob_sum is None else (prob_sum + prob)
        prob_avg = prob_sum / len(variants)
        topk_prob, topk_idx = torch.topk(prob_avg, k=topk)
        topk_prob = topk_prob.cpu().numpy().flatten().tolist()
        topk_idx = topk_idx.cpu().numpy().flatten().tolist()
        return [(self.idx_to_class[i], float(p)) for i,p in zip(topk_idx, topk_prob)]

    def predict_path(self, path, topk=5):
        img = imread_rgb(path)
        return self.predict_image(img, topk=topk)

    def predict_multicrop(self, img_rgb, topk=5, tta: str = "fast"):
        """5-crop averaging (center + 4 corners) to improve robustness without a detector."""
        h, w = img_rgb.shape[:2]
        # define crop boxes as ratios
        s = 0.75  # crop size ratio
        cw, ch = int(w * s), int(h * s)
        boxes = [
            ( (w - cw)//2, (h - ch)//2, (w + cw)//2, (h + ch)//2 ),  # center
            ( 0, 0, cw, ch ),                                        # top-left
            ( w - cw, 0, w, ch ),                                    # top-right
            ( 0, h - ch, cw, h ),                                    # bottom-left
            ( w - cw, h - ch, w, h ),                                # bottom-right
        ]
        prob_sum = None
        use_cuda = (self.device.type == "cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            for (x1,y1,x2,y2) in boxes:
                x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img_rgb[y1:y2, x1:x2, :]
                # reuse predict_image pipeline variant with tta
                variants = [crop]
                if tta and tta != "off":
                    variants.append(np.ascontiguousarray(crop[:, ::-1, :]))
                    if tta == "full":
                        variants.append(np.ascontiguousarray(crop[::-1, :, :]))
                for v in variants:
                    x = self.tf(image=v)["image"].unsqueeze(0).to(self.device).to(memory_format=torch.channels_last)
                    logits = self.model(x)
                    prob = torch.softmax(logits, dim=1)
                    prob_sum = prob if prob_sum is None else (prob_sum + prob)
        if prob_sum is None:
            return self.predict_image(img_rgb, topk=topk, tta=tta)
        prob_avg = prob_sum / (len(boxes) * (2 if tta=="fast" else (3 if tta=="full" else 1)))
        topk_prob, topk_idx = torch.topk(prob_avg, k=topk)
        topk_prob = topk_prob.cpu().numpy().flatten().tolist()
        topk_idx = topk_idx.cpu().numpy().flatten().tolist()
        return [(self.idx_to_class[i], float(p)) for i,p in zip(topk_idx, topk_prob)]
