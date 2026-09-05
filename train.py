#!/usr/bin/env python3
"""Trains the height regressor. Mirrors satellite_heights.ipynb -- same
hyperparameters, same split, same schedule -- but runnable as a script.

    python3 train.py --data-dir /path/to/tiles --csv /path/to/metadata.csv

The dataset (tiles + a metadata CSV with filename/height/status columns) is
not distributed with this repository; see the README.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from astana_heights import (SatelliteHeightDataset, build_model, build_transforms,
                            format_report, load_metadata, make_splits, set_trainable)
from astana_heights.config import (BATCH_SIZE, EPOCHS, FREEZE_BACKBONE_EPOCHS,
                                    HUBER_DELTA, LR, NUM_WORKERS, PATIENCE, SEED,
                                    WEIGHT_DECAY)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, _, raw_h in loader:
            out = model(imgs.to(device))
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(raw_h.numpy().tolist())
    return preds, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="directory of tile images")
    ap.add_argument("--csv", required=True, help="metadata CSV (filename, height, status)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--out", default="best_model.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    df = load_metadata(args.csv, args.data_dir)
    print(f"Samples: {len(df)}, Height: {df['height'].min():.1f}m - "
          f"{df['height'].max():.1f}m, Median: {df['height'].median():.1f}m")
    train_df, val_df, test_df = make_splits(df, args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_tf, eval_tf = build_transforms()
    mk = lambda d, tf, sh: DataLoader(
        SatelliteHeightDataset(d, args.data_dir, tf),
        batch_size=args.batch_size, shuffle=sh, num_workers=args.num_workers)
    train_loader = mk(train_df, train_tf, True)
    val_loader = mk(val_df, eval_tf, False)
    test_loader = mk(test_df, eval_tf, False)

    model = build_model(pretrained=True).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.HuberLoss(delta=HUBER_DELTA, reduction="mean")

    # Head-only warmup: the head is random, so letting its gradients reach the
    # pretrained backbone immediately would scramble good features.
    set_trainable(model.features, False)
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.lr},
        {"params": model.features.parameters(), "lr": args.lr / 10},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_mae, best_state, patience_counter = float("inf"), None, 0

    for epoch in range(args.epochs):
        if epoch == FREEZE_BACKBONE_EPOCHS:
            set_trainable(model.features, True)
            optimizer = torch.optim.AdamW([
                {"params": model.head.parameters(), "lr": args.lr},
                {"params": model.features.parameters(), "lr": args.lr / 10},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6)
            print(f"\n[Epoch {epoch}] Unfreezing backbone")

        model.train()
        tl = tm = n = 0
        for imgs, target, raw_h in tqdm(train_loader, desc=f"Train Ep {epoch+1}", leave=False):
            imgs, target = imgs.to(device), target.float().to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(imgs)
            tm += torch.abs(pred - raw_h.to(device)).sum().item()
            n += len(imgs)

        model.eval()
        vl = vm = nv = 0
        with torch.no_grad():
            for imgs, target, raw_h in val_loader:
                imgs, target = imgs.to(device), target.float().to(device)
                pred = model(imgs)
                vl += criterion(pred, target).item() * len(imgs)
                vm += torch.abs(pred - raw_h.to(device)).sum().item()
                nv += len(imgs)

        tl, tm, vl, vm = tl / n, tm / n, vl / nv, vm / nv
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:3d} | Train Loss: {tl:.2f} | Train MAE: {tm:.2f}m | "
                  f"Val Loss: {vl:.2f} | Val MAE: {vm:.2f}m | LR: {scheduler.get_last_lr()[0]:.6f}")

        if vm < best_val_mae:
            best_val_mae, patience_counter = vm, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}. Best Val MAE: {best_val_mae:.2f}m")
                break

    print(f"\nBest Validation MAE: {best_val_mae:.2f}m")

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, args.out)
        print(f"Saved {args.out}")

    preds, targets = evaluate(model, test_loader, device)
    print("\n" + format_report(preds, targets))


if __name__ == "__main__":
    main()
