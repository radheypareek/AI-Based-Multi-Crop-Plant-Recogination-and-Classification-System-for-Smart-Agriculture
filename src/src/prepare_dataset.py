import os, shutil, argparse, random
from pathlib import Path
from tqdm import tqdm

def copy_images(file_list, out_root):
    for src in tqdm(file_list):
        rel = "/".join(Path(src).parts[-2:])
        dst = Path(out_root) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def main(args):
    raw_root = Path(args.input)
    classes = sorted([d.name for d in raw_root.iterdir() if d.is_dir()])
    all_files = []
    for c in classes:
        files = list((raw_root/c).glob("*.*"))
        for f in files: all_files.append(str(f))

    # stratified split per class
    train, val, test = [], [], []
    for c in classes:
        imgs = list((raw_root/c).glob("*.*"))
        random.shuffle(imgs)
        n = len(imgs)
        n_val = int(n*args.val_split)
        n_test = int(n*args.test_split)
        val.extend(imgs[:n_val])
        test.extend(imgs[n_val:n_val+n_test])
        train.extend(imgs[n_val+n_test:])

    for split in ["train","val","test"]:
        (Path(args.output)/split).mkdir(parents=True, exist_ok=True)

    copy_images(train, Path(args.output)/"train")
    copy_images(val, Path(args.output)/"val")
    copy_images(test, Path(args.output)/"test")
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--test_split", type=float, default=0.1)
    args = ap.parse_args()
    main(args)
