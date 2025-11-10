import argparse, torch
import timm

def main(args):
    ckpt = torch.load(args.weights, map_location="cpu")
    class_to_idx = ckpt['class_to_idx']
    model = timm.create_model(args.model_name, pretrained=False, num_classes=len(class_to_idx))
    model.load_state_dict(ckpt['model']); model.eval()
    dummy = torch.randn(1,3,args.img_size,args.img_size)
    torch.onnx.export(model, dummy, args.output, input_names=["input"], output_names=["logits"], opset_version=13)
    print("Exported ONNX to", args.output)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_name", default="efficientnet_b0")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    main(args)
