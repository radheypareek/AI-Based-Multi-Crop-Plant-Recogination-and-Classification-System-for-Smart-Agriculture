import argparse, torch, cv2
from pathlib import Path
from src.dataset import imread_rgb
from src.transforms import val_transforms
from src.common import ensure_dir
import numpy as np
import timm

def overlay_heatmap(img_rgb, mask, alpha=0.4):
    heatmap = cv2.applyColorMap((255*mask).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (alpha*heatmap + (1-alpha)*img_rgb).astype(np.uint8)
    return blended

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    class_to_idx = ckpt['class_to_idx']
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_to_idx))
    model.load_state_dict(ckpt['model']); model.to(device); model.eval()

    t_val = val_transforms(args.img_size, [0.485,0.456,0.406], [0.229,0.224,0.225])

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except:
        raise ImportError("Install grad-cam package")

    target_layers = [model.get_layer("blocks.6.2.conv_pw")] if hasattr(model, "get_layer") else [list(model.modules())[-2]]
    out_dir = Path(args.output); ensure_dir(out_dir)

    for p in Path(args.input).glob("*.*"):
        img = imread_rgb(p)
        x = t_val(image=img)["image"].unsqueeze(0).to(device)
        with GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type=="cuda")) as cam:
            grayscale_cam = cam(input_tensor=x, targets=None)
        overlay = overlay_heatmap(cv2.resize(img, (args.img_size, args.img_size)), grayscale_cam)
        cv2.imwrite(str(out_dir/f"{p.stem}_cam.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Saved Grad-CAM overlays to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="reports/gradcam")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    main(args)
