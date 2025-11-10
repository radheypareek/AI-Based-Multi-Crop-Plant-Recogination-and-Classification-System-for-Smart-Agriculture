# src/training/train.py (resume + speed)
import os, argparse, math, time, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.common import set_seed, ensure_dir
from src.transforms import train_transforms, val_transforms
from src.dataset import ImageFolderAlb
from src.build import build_model, LabelSmoothingCE

def get_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.scheduler.name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs * steps_per_epoch))
    return None

def mixup_data(x, y, alpha=0.2):
    if not alpha or alpha <= 0: return x, (y, y), 1.0, False
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    lam = max(0.0, min(1.0, lam))
    if lam < 1e-3 or lam > 1.0 - 1e-3: return x, (y, y), 1.0, False
    b = x.size(0)
    idx = torch.randperm(b, device=x.device)
    return lam * x + (1 - lam) * x[idx], (y, y[idx]), lam, True

def save_state(path, model, optimizer, scheduler, scaler, epoch, step, best_metric, class_to_idx, cfg, rng_state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "class_to_idx": class_to_idx,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "rng": rng_state,
    }, path)

def load_state(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights. Missing: {missing}, Unexpected: {unexpected}")
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best_metric", 0.0), ckpt.get("class_to_idx")

def main(args):
    cfg = OmegaConf.load(args.config)
    # Optional overrides from CLI for quick experiments
    if hasattr(args, "epochs") and args.epochs and args.epochs > 0:
        cfg.epochs = int(args.epochs)
    set_seed(cfg.seed)

    use_cuda = torch.cuda.is_available() and cfg.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    mean, std = cfg.data.mean, cfg.data.std
    t_train = train_transforms(cfg.img_size, mean, std, cfg.augment.color_jitter, cfg.augment.randaugment)
    t_val   = val_transforms(cfg.img_size, mean, std)

    train_root = Path(cfg.data.root)/cfg.data.train_dir
    val_root   = Path(cfg.data.root)/cfg.data.val_dir

    # Build train dataset first to fix mapping
    train_ds = ImageFolderAlb(train_root, t_train)
    val_ds   = ImageFolderAlb(val_root, t_val, class_to_idx=train_ds.class_to_idx)

    # If limiting number of training samples, wrap in a Subset without touching the dataset on disk
    limit_train = getattr(args, "limit_train", 0) or 0
    if limit_train and limit_train > 0:
        from torch.utils.data import Subset
        total = len(train_ds)
        k = min(int(limit_train), total)
        indices = random.sample(range(total), k)
        train_ds = Subset(train_ds, indices)

    # Resolve base dataset for class mapping if using a Subset
    try:
        from torch.utils.data import Subset as _Subset
        base_train_ds = train_ds.dataset if isinstance(train_ds, _Subset) else train_ds
    except Exception:
        base_train_ds = train_ds

    ncls = len(base_train_ds.class_to_idx)
    assert ncls == cfg.num_classes, f"class count {ncls} != cfg.num_classes={cfg.num_classes}"

    # Faster DataLoader settings
    # - On Windows: try 2 workers (stable speedup vs 0); fall back to 0 if it errors in your environment
    # - On Linux: use up to 8 workers depending on CPU
    num_workers = 2 if os.name == 'nt' else min(8, os.cpu_count() or 4)
    pin_memory = True if device.type == "cuda" else False
    persistent = True if num_workers > 0 else False
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)

    model = build_model(cfg.model_name, cfg.num_classes, pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    # Freeze backbone for first N epochs if configured
    freeze_backbone_epochs = int(getattr(cfg, "freeze_backbone_epochs", 0) or 0)
    def _mark_trainable_for_warmup(m):
        classifier_keywords = ["classifier", "fc", "head", "heads", "classif", "last_linear"]
        for n, p in m.named_parameters():
            if any(k in n for k in classifier_keywords):
                p.requires_grad = True
            else:
                p.requires_grad = False
    if freeze_backbone_epochs > 0:
        _mark_trainable_for_warmup(model)
    can_compile = False
    if torch.cuda.is_available():
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        can_compile = (cc_major, cc_minor) >= (7, 0)
    if can_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile disabled: {e}")

    criterion = LabelSmoothingCE(cfg.label_smoothing)
    # only optimize trainable parameters (some may be frozen during warmup)
    def make_optimizer():
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay,
                                 fused=True if use_cuda else False)
    optimizer = make_optimizer()
    steps_per_epoch = max(1, len(train_loader))
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch)

    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=use_cuda)
    accum_steps = getattr(cfg, "accum_steps", 1)

    ckpt_dir = Path(cfg.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = 0.0
    start_epoch = 0
    global_step = 0

    # If --resume provided, use it; else auto-pick latest "last_epoch*.pt" if present
    resume_path = args.resume
    if not resume_path:
        last_ckpts = sorted(ckpt_dir.glob("last_epoch*.pt"))
        if last_ckpts:
            resume_path = str(last_ckpts[-1])

    # Guard: skip resume if checkpoint was trained with different model/num_classes
    if resume_path and Path(resume_path).is_file():
        try:
            meta = torch.load(resume_path, map_location="cpu")
            prev_cfg = meta.get("cfg", {}) or {}
            prev_model = prev_cfg.get("model_name")
            prev_ncls = None
            if "class_to_idx" in meta and isinstance(meta["class_to_idx"], dict):
                prev_ncls = len(meta["class_to_idx"])  # robust even if cfg missing
            # If architecture or class count changed, do NOT resume
            if (prev_model and prev_model != cfg.model_name) or (prev_ncls and prev_ncls != cfg.num_classes):
                print(f"[INFO] Skip resume from {resume_path} due to config mismatch: prev_model={prev_model}, prev_ncls={prev_ncls}, current_model={cfg.model_name}, current_ncls={cfg.num_classes}")
                resume_path = ""
        except Exception as e:
            print(f"[WARN] Could not inspect checkpoint {resume_path}: {e}. Proceeding without resume.")
            resume_path = ""

    if resume_path and Path(resume_path).is_file():
        print(f"Resuming from {resume_path}")
        start_epoch, global_step, best_metric, c2i = load_state(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        # Optional: validate class_to_idx matches
        if c2i and hasattr(train_ds, "class_to_idx") and c2i != train_ds.class_to_idx:
            raise RuntimeError("class_to_idx mismatch between checkpoint and current dataset.")


    val_every = int(getattr(cfg, "val_every", 1) or 1)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", ncols=100)

        for step, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            if y.min() < 0 or y.max() >= cfg.num_classes:
                raise ValueError(f"Label out of range: min={int(y.min())}, max={int(y.max())}, expected [0,{cfg.num_classes-1}]")

            do_mix = (cfg.augment.mixup > 0.0 or cfg.augment.cutmix > 0.0)
            alpha = max(cfg.augment.mixup, cfg.augment.cutmix) if do_mix else 0.0
            if do_mix and alpha > 0.0:
                x, (ya, yb), lam, ok = mixup_data(x, y, alpha=alpha)
            else:
                ya, yb, lam, ok = y, y, 1.0, False

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(x)
                if ok:
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                else:
                    loss = criterion(logits, y)
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

            running_loss += loss.item() * accum_steps
            if not ok:
                with torch.no_grad():
                    pred = logits.argmax(1)
                    batch_acc = (pred == y).float().mean().item()
                    running_acc = 0.9 * running_acc + 0.1 * batch_acc

            global_step += 1
            pbar.set_postfix(loss=f"{running_loss/(step+1):.4f}", acc=f"{running_acc:.3f}")

            # Periodic checkpoint every N steps (e.g., 500)
            ckpt_every_steps = getattr(cfg, "ckpt_every_steps", 0)
            if ckpt_every_steps > 0 and (global_step % ckpt_every_steps == 0):
                save_state(str(ckpt_dir / f"epoch{epoch:03d}_step{global_step:06d}.pt"),
                        model, optimizer, scheduler, scaler, epoch, global_step, best_metric,
                        base_train_ds.class_to_idx, cfg, torch.get_rng_state())

        # Unfreeze backbone after warmup epochs and rebuild optimizer/scheduler once
        if freeze_backbone_epochs > 0 and epoch + 1 == freeze_backbone_epochs:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = make_optimizer()
            scheduler = get_scheduler(optimizer, cfg, steps_per_epoch)

        # Validation (every N epochs or on final epoch)
        do_validate = ((epoch + 1) % val_every == 0) or (epoch + 1 == cfg.epochs)
        val_acc = -1.0
        if do_validate:
            model.eval()
            correct1 = 0; total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                        logits = model(x)
                    pred = logits.argmax(1)
                    correct1 += (pred == y).sum().item()
                    total += y.size(0)
            val_acc = correct1 / max(1, total)
            print(f"Val Acc: {val_acc:.4f}")

        # Save best and an end-of-epoch checkpoint
        # Save "best.pt" if improved
        if val_acc >= 0 and val_acc > best_metric:
            best_metric = val_acc
            save_state(str(ckpt_dir / "best.pt"),
                    model, optimizer, scheduler, scaler, epoch, global_step, best_metric,
                    base_train_ds.class_to_idx, cfg, torch.get_rng_state())

        # Always update rolling "last_epochXXX.pt" for auto-resume
        save_state(str(ckpt_dir / f"last_epoch{epoch:03d}.pt"),
                model, optimizer, scheduler, scaler, epoch, global_step, best_metric,
                base_train_ds.class_to_idx, cfg, torch.get_rng_state())

        # Early stopping
        if hasattr(cfg, "early_stopping") and cfg.early_stopping:
            # Implement patience based on best_metric if desired
            pass

    print("Training complete. Best Val Acc:", best_metric)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", type=str, default="")
    # Fast-experiment controls (do not modify dataset on disk)
    ap.add_argument("--epochs", type=int, default=0, help="Override number of epochs")
    ap.add_argument("--limit_train", type=int, default=0, help="Limit number of training samples")
    args = ap.parse_args()
    main(args)
