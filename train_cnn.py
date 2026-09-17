"""
train_cnn.py — train the lightweight CNN scorer on bridge.py's crop manifest.

Run on your own machine/Colab (needs torch+torchvision; not available in the
sandbox this was authored in, so the manifest/split/weighting logic was
pre-validated separately via manifest_utils.py -- only the actual model/
training loop below is untested end-to-end. Watch the first epoch's printed
loss for anything obviously broken (NaN, not decreasing at all) before
trusting a full run).

ARCHITECTURE: deliberately small (edge-deployment requirement from the
project brief) -- 4 conv blocks with increasing channels, global average
pool (avoids a large flatten+FC layer, which is usually where most of a
small CNN's parameter budget silently goes), single FC classification head.

USAGE:
  python3 train_cnn.py --manifest bridge_out/manifest.csv --out model.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from manifest_utils import (load_manifest, grouped_train_val_split,
                             build_label_map, class_weights, class_counts,
                             matched_vs_fallback)


class CropDataset(Dataset):
    def __init__(self, rows, label_map, crop_size=64, augment=False):
        self.rows = rows
        self.label_map = label_map
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(row["crop_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # crop file missing/corrupt -- fail loudly at load time rather
            # than silently feeding zeros into training
            raise FileNotFoundError(f"could not read crop: {row['crop_path']}")
        img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 0)
            k = np.random.choice([0, 1, 2, 3])
            if k:
                img = np.rot90(img, k).copy()

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        label = self.label_map[row["label"]]
        return tensor, label


class SonarCNN(nn.Module):
    """Small enough for edge/onboard-drone inference: ~4 conv blocks,
    global average pool instead of a big FC layer. Parameter count printed
    at construction time so you can see the actual edge-deployment budget."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def evaluate(model, loader, device, n_classes, idx_to_label):
    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1

    print("\nconfusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{idx_to_label[i]:>12}" for i in range(n_classes))
    print(" " * 12 + header)
    for i in range(n_classes):
        row_str = "".join(f"{confusion[i, j]:>12}" for j in range(n_classes))
        print(f"{idx_to_label[i]:>12}{row_str}")

    print("\nper-class precision/recall:")
    for i in range(n_classes):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        flag = "  <-- weakest" if recall < 0.5 else ""
        print(f"  {idx_to_label[i]:>12}: precision={precision:.3f}  recall={recall:.3f}{flag}")

    acc = np.trace(confusion) / confusion.sum() if confusion.sum() else float("nan")
    print(f"\noverall accuracy: {acc:.3f}")
    return confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rows = load_manifest(args.manifest)
    print(f"loaded {len(rows)} rows. class counts: {class_counts(rows)}")
    print("matched vs fallback per class:")
    for label, d in matched_vs_fallback(rows).items():
        total = sum(d.values())
        matched_pct = 100 * d.get("cfar_matched", 0) / total if total else float("nan")
        print(f"  {label:>12}: {d}  ({matched_pct:.1f}% real CFAR matches)")

    if "background" not in class_counts(rows):
        print("\n[WARNING] no 'background' class in this manifest -- the CNN "
              "will never learn to reject CFAR's false-alarm proposals, which "
              "will be most of what it sees at real inference time. Re-check "
              "bridge.py's --neg-per-image / background/ output before "
              "trusting this model downstream.")

    label_map = build_label_map(rows)
    idx_to_label = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    print(f"\nlabel map: {label_map}")

    train_rows, val_rows = grouped_train_val_split(rows, val_frac=args.val_frac, seed=args.seed)
    print(f"train: {len(train_rows)} rows, val: {len(val_rows)} rows (grouped by source image, no leakage)")

    weights = class_weights(train_rows, label_map)
    print(f"class weights (train set): {[f'{w:.3f}' for w in weights]}")

    train_ds = CropDataset(train_rows, label_map, crop_size=args.crop_size, augment=True)
    val_ds = CropDataset(val_rows, label_map, crop_size=args.crop_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SonarCNN(n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,} (edge-deployment budget check)")

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0

        print(f"epoch {epoch:>3}/{args.epochs}  train_loss={total_loss/max(n_batches,1):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "label_map": label_map,
                        "crop_size": args.crop_size}, args.out)

    print(f"\nbest val_acc: {best_val_acc:.4f}, saved to {args.out}")
    print("\nFinal evaluation on val set (best checkpoint):")
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    evaluate(model, val_loader, device, n_classes, idx_to_label)


if __name__ == "__main__":
    main()
