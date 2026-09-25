#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation disque optique / cupule sur Drishti-GS1, et calcul du CDR
=======================================================================

Contrairement à la classification (glaucome/normal), on prédit ici, pour
chaque pixel de l'image, s'il appartient au disque optique et/ou à la
cupule. On en déduit le CDR (Cup-to-Disc Ratio), un critère clinique
classique : un CDR élevé est un signe de glaucome.

Arborescence attendue :
    <root>/
      Images/
        glaucoma/*.png|jpg
        normal/*.png|jpg
      Test_GT/
        drishtiGS_XXX/
          SoftMap/
            drishtiGS_XXX_cupsegSoftmap.png   <- masque de la cupule (0-255)
            drishtiGS_XXX_ODsegSoftmap.png    <- masque du disque (0-255)
          drishtiGS_XXX_cdrValues.txt         <- CDR de 4 experts, ex. "0.85 0.82 0.80 0.82"

Seules les images ayant un dossier Test_GT correspondant sont utilisées.

Installation : pip install torch torchvision scikit-learn pandas matplotlib pillow

Exemple :
    python drishti_segmentation.py --root "data/drishti gs" --epochs 40 --img_size 256
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Chargement des cas (image + masques + CDR de référence)
# --------------------------------------------------------------------------- #
def load_cases(root: str) -> pd.DataFrame:
    root = Path(root)
    images_dir = root / "Images"
    gt_dir = root / "Test_GT"
    if not images_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Attendu {images_dir} et {gt_dir}")

    # nom_sans_extension -> chemin de l'image (glaucoma ou normal)
    img_by_key = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            img_by_key[p.stem.lower()] = p

    rows = []
    for case_dir in sorted(gt_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        key = case_dir.name.lower()
        img_path = img_by_key.get(key)
        if img_path is None:
            print(f"[!] Pas d'image trouvée pour {case_dir.name}, cas ignoré.")
            continue

        soft = case_dir / "SoftMap"
        cup_path = next(soft.glob("*cupsegSoftmap*"), None)
        od_path = next(soft.glob("*ODsegSoftmap*"), None)
        if cup_path is None or od_path is None:
            print(f"[!] SoftMap manquant pour {case_dir.name}, cas ignoré.")
            continue

        cdr_file = next(case_dir.glob("*cdrValues*"), None)
        cdr_ref = np.nan
        if cdr_file is not None:
            values = [float(v) for v in cdr_file.read_text().split()]
            cdr_ref = float(np.mean(values))

        label = 1 if "glaucoma" in str(img_path.parent).lower() else 0
        rows.append({
            "case": case_dir.name,
            "image": str(img_path),
            "cup_mask": str(cup_path),
            "od_mask": str(od_path),
            "cdr_ref": cdr_ref,
            "label": label,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("Aucun cas complet (image + masques) trouvé.")
    print(f"[Drishti-GS] {len(df)} cas complets | glaucome={int(df.label.sum())} "
          f"| normal={int((df.label == 0).sum())} | CDR ref. moyen={df.cdr_ref.mean():.3f}")
    return df


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SegDataset(Dataset):
    """Renvoie (image [3,H,W], masque [2,H,W] = [OD, cup]), même augmentation
    (flip/rotation) appliquée à l'image ET aux masques ensemble."""

    def __init__(self, df: pd.DataFrame, size: int, augment: bool):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.image).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        od = Image.open(r.od_mask).convert("L").resize((self.size, self.size), Image.NEAREST)
        cup = Image.open(r.cup_mask).convert("L").resize((self.size, self.size), Image.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        od = (np.asarray(od, dtype=np.float32) / 255.0) > 0.5
        cup = (np.asarray(cup, dtype=np.float32) / 255.0) > 0.5

        if self.augment:
            if random.random() < 0.5:  # flip horizontal
                img, od, cup = img[:, ::-1].copy(), od[:, ::-1].copy(), cup[:, ::-1].copy()
            if random.random() < 0.5:  # flip vertical
                img, od, cup = img[::-1, :].copy(), od[::-1, :].copy(), cup[::-1, :].copy()
            k = random.choice([0, 1, 2, 3])  # rotation de 0/90/180/270°
            if k:
                img = np.rot90(img, k).copy()
                od = np.rot90(od, k).copy()
                cup = np.rot90(cup, k).copy()

        img_t = torch.from_numpy(img.transpose(2, 0, 1))                       # [3,H,W]
        mask_t = torch.from_numpy(np.stack([od, cup]).astype(np.float32))      # [2,H,W]
        return img_t, mask_t, r.case, float(r.cdr_ref)


# --------------------------------------------------------------------------- #
# Modèle : petit U-Net
# --------------------------------------------------------------------------- #
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net compact : 4 niveaux d'encodeur/décodeur, 2 canaux de sortie (OD, cupule)."""

    def __init__(self, out_channels: int = 2, base: int = 32):
        super().__init__()
        self.e1 = conv_block(3, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.e4 = conv_block(base * 4, base * 8)
        self.bottleneck = conv_block(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.d4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)  # logits, [B,2,H,W]


# --------------------------------------------------------------------------- #
# Perte et métriques de segmentation
# --------------------------------------------------------------------------- #
def dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(torch.sigmoid(logits), target)
    return bce + dice


@torch.no_grad()
def dice_score(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    dims = (0, 2, 3)
    inter = (pred_bin * target).sum(dims)
    union = pred_bin.sum(dims) + target.sum(dims)
    return ((2 * inter + eps) / (union + eps)).cpu().numpy()  # [OD, cup]


@torch.no_grad()
def iou_score(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    dims = (0, 2, 3)
    inter = (pred_bin * target).sum(dims)
    union = pred_bin.sum(dims) + target.sum(dims) - inter
    return ((inter + eps) / (union + eps)).cpu().numpy()  # [OD, cup]


def vertical_diameter(mask: np.ndarray) -> float:
    """Diamètre vertical d'un masque binaire 2D (nb de lignes contenant au moins un pixel actif)."""
    rows = np.where(mask.any(axis=1))[0]
    return float(rows.max() - rows.min() + 1) if len(rows) else 0.0


def compute_cdr(od_mask: np.ndarray, cup_mask: np.ndarray) -> float:
    """CDR clinique standard = diamètre vertical de la cupule / diamètre vertical du disque."""
    od_d = vertical_diameter(od_mask)
    return float(vertical_diameter(cup_mask) / od_d) if od_d > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Entraînement / évaluation
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    all_dice, all_iou, n = np.zeros(2), np.zeros(2), 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = combined_loss(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            pred_bin = (torch.sigmoid(logits) > 0.5).float()
            all_dice += dice_score(pred_bin, y) * x.size(0)
            all_iou += iou_score(pred_bin, y) * x.size(0)
            n += x.size(0)
    return total_loss / n, all_dice / n, all_iou / n  # [OD, cup] pour dice/iou


@torch.no_grad()
def evaluate_cdr(model, loader, device):
    model.eval()
    rows = []
    for x, y, cases, cdr_ref in loader:
        x = x.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()  # [B,2,H,W]
        for i in range(len(cases)):
            od_pred, cup_pred = pred[i, 0], pred[i, 1]
            od_true, cup_true = y[i, 0].numpy(), y[i, 1].numpy()
            rows.append({
                "case": cases[i],
                "cdr_ref": float(cdr_ref[i]),
                "cdr_pred": compute_cdr(od_pred, cup_pred),
                "cdr_true_mask": compute_cdr(od_true, cup_true),  # CDR recalculé sur le masque GT (référence de méthode)
            })
    return pd.DataFrame(rows)


def save_examples(model, df: pd.DataFrame, size: int, device, out_dir: Path, n: int = 8):
    out = out_dir / "examples"
    out.mkdir(parents=True, exist_ok=True)
    ds = SegDataset(df.sample(min(n, len(df)), random_state=0), size, augment=False)
    for i in range(len(ds)):
        x, y, case, cdr_ref = ds[i]
        with torch.no_grad():
            pred = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0].cpu().numpy()
        img = x.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img)
        ax[0].set_title(f"{case}\nCDR référence={cdr_ref:.2f}")
        ax[1].imshow(img)
        ax[1].contour(y[0].numpy(), colors="lime", linewidths=1.5)   # OD réel
        ax[1].contour(y[1].numpy(), colors="red", linewidths=1.5)    # cupule réelle
        ax[1].set_title("Vérité terrain (vert=disque, rouge=cupule)")
        ax[2].imshow(img)
        ax[2].contour(pred[0] > 0.5, colors="lime", linewidths=1.5)
        ax[2].contour(pred[1] > 0.5, colors="red", linewidths=1.5)
        pred_cdr = compute_cdr(pred[0] > 0.5, pred[1] > 0.5)
        ax[2].set_title(f"Prédiction (CDR calculé={pred_cdr:.2f})")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{case}.png", dpi=120)
        plt.close(fig)


def save_predicted_masks(model, df: pd.DataFrame, size: int, device, out_dir: Path):
    """Enregistre, pour chaque cas, le masque OD, le masque cupule (PNG binaires
    noir/blanc, même taille que l'image d'entrée du modèle) et une image de
    synthèse (vert=disque, rouge=cupule) sur fond noir."""
    out = out_dir / "masks"
    out.mkdir(parents=True, exist_ok=True)
    ds = SegDataset(df, size, augment=False)
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x, _, case, _ = ds[i]
            pred = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0].cpu().numpy()
            od_mask = (pred[0] > 0.5).astype(np.uint8) * 255
            cup_mask = (pred[1] > 0.5).astype(np.uint8) * 255

            Image.fromarray(od_mask).save(out / f"{case}_OD_mask.png")
            Image.fromarray(cup_mask).save(out / f"{case}_cup_mask.png")

            overlay = np.zeros((size, size, 3), dtype=np.uint8)
            overlay[..., 1] = od_mask   # vert = disque
            overlay[..., 0] = cup_mask  # rouge = cupule
            Image.fromarray(overlay).save(out / f"{case}_overlay_mask.png")
    print(f"Masques enregistrés dans : {out.resolve()} ({len(ds)} cas)")


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Segmentation disque/cupule + CDR sur Drishti-GS")
    ap.add_argument("--root", required=True, help='ex. "data/drishti gs"')
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs/drishti_seg")
    ap.add_argument("--masks_for", choices=["test", "all"], default="all",
                    help="Pour quelles images sauvegarder le masque prédit à la fin : "
                         "'test' (les 11 cas de test) ou 'all' (les 51 cas).")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    df = load_cases(args.root)
    # 51 cas seulement : 70/15/15 environ, stratifié sur le label glaucome/normal
    trainval, test = train_test_split(df, test_size=0.2, stratify=df.label, random_state=args.seed)
    train, val = train_test_split(trainval, test_size=0.2, stratify=trainval.label, random_state=args.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s}: {len(part):2d} cas | glaucome={int(part.label.sum())}")

    train_loader = DataLoader(SegDataset(train, args.img_size, augment=True),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SegDataset(val, args.img_size, augment=False),
                            batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SegDataset(test, args.img_size, augment=False),
                             batch_size=args.batch_size, shuffle=False)

    model = UNet(out_channels=2, base=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice, best_epoch, history = -1.0, 0, []
    ckpt = out_dir / "best_unet.pt"
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice, _ = run_epoch(model, train_loader, device, optimizer)
        scheduler.step()
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, device, optimizer=None)
        mean_val_dice = val_dice.mean()
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_dice_od": float(val_dice[0]), "val_dice_cup": float(val_dice[1])})
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={tr_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val Dice OD={val_dice[0]:.3f} cup={val_dice[1]:.3f}")
        if mean_val_dice > best_dice:
            best_dice, best_epoch = mean_val_dice, epoch
            torch.save(model.state_dict(), ckpt)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping (meilleure epoch : {best_epoch}, Dice moyen={best_dice:.3f})")
            break

    # Évaluation finale sur le test, avec le meilleur modèle
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_dice, test_iou = run_epoch(model, test_loader, device, optimizer=None)
    cdr_df = evaluate_cdr(model, test_loader, device)
    cdr_mae = float((cdr_df.cdr_pred - cdr_df.cdr_ref).abs().mean())

    results = {
        "dice_OD": float(test_dice[0]), "dice_cup": float(test_dice[1]),
        "iou_OD": float(test_iou[0]), "iou_cup": float(test_iou[1]),
        "cdr_mae_vs_reference": cdr_mae,
        "history": history,
    }
    print("\n=== Résultats sur le jeu de test (segmentation) ===")
    print(f"Dice  : OD={test_dice[0]:.3f} | cupule={test_dice[1]:.3f}")
    print(f"IoU   : OD={test_iou[0]:.3f} | cupule={test_iou[1]:.3f}")
    print(f"Erreur moyenne absolue du CDR (vs experts) : {cdr_mae:.3f}")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    cdr_df.to_csv(out_dir / "cdr_predictions.csv", index=False)
    save_examples(model, test, args.img_size, device, out_dir)
    mask_targets = df if args.masks_for == "all" else test
    save_predicted_masks(model, mask_targets, args.img_size, device, out_dir)
    print(f"\nRésultats enregistrés dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
