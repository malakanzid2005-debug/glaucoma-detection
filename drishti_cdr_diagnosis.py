#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic glaucome / normal à partir du CDR (Drishti-GS)
===========================================================

Approche "règle clinique", différente du CNN de classification directe :
  1) Un U-Net segmente le disque optique et la cupule sur chaque image.
  2) On calcule le CDR (Cup-to-Disc Ratio) à partir de ces deux masques.
  3) Une image est déclarée "glaucome suspecté" si son CDR dépasse un seuil,
     comme le ferait un ophtalmologue avec la règle "CDR > 0.5 (ou 0.6) = suspect".
  4) On compare ce diagnostic au vrai label (dossier Images/glaucoma|normal).

Le seuil de décision est choisi sur train+val (jamais sur le test), en
maximisant l'indice de Youden sur le CDR utilisé comme score continu — la
même logique que pour le seuil de probabilité en classification directe.

Arborescence attendue (identique à drishti_segmentation.py) :
    <root>/
      Images/
        glaucoma/*.png|jpg
        normal/*.png|jpg
      Test_GT/
        drishtiGS_XXX/
          SoftMap/
            drishtiGS_XXX_cupsegSoftmap.png
            drishtiGS_XXX_ODsegSoftmap.png
          drishtiGS_XXX_cdrValues.txt   (facultatif, CDR des experts, à titre de repère)

Exemple :
    python drishti_cdr_diagnosis.py --root "data/drishti gs" --epochs 40 --img_size 256
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Chargement des cas
# --------------------------------------------------------------------------- #
def load_cases(root: str) -> pd.DataFrame:
    root = Path(root)
    images_dir = root / "Images"
    gt_dir = root / "Test_GT"
    if not images_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Attendu {images_dir} et {gt_dir}")

    img_by_key = {p.stem.lower(): p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXT}

    rows = []
    for case_dir in sorted(gt_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        key = case_dir.name.lower()
        img_path = img_by_key.get(key)
        if img_path is None:
            continue
        soft = case_dir / "SoftMap"
        cup_path = next(soft.glob("*cupsegSoftmap*"), None)
        od_path = next(soft.glob("*ODsegSoftmap*"), None)
        if cup_path is None or od_path is None:
            continue

        cdr_file = next(case_dir.glob("*cdrValues*"), None)
        cdr_expert = np.nan
        if cdr_file is not None:
            values = [float(v) for v in cdr_file.read_text().split()]
            cdr_expert = float(np.mean(values))

        label = 1 if "glaucoma" in str(img_path.parent).lower() else 0
        rows.append({
            "case": case_dir.name, "image": str(img_path),
            "cup_mask": str(cup_path), "od_mask": str(od_path),
            "cdr_expert": cdr_expert, "label": label,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("Aucun cas complet (image + masques) trouvé.")
    print(f"[Drishti-GS] {len(df)} cas | glaucome={int(df.label.sum())} "
          f"| normal={int((df.label == 0).sum())}")
    return df


# --------------------------------------------------------------------------- #
# Dataset (image + masques ; le label sert seulement à l'évaluation finale,
# JAMAIS à l'entraînement du U-Net, qui apprend uniquement à segmenter)
# --------------------------------------------------------------------------- #
class SegDataset(Dataset):
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
            if random.random() < 0.5:
                img, od, cup = img[:, ::-1].copy(), od[:, ::-1].copy(), cup[:, ::-1].copy()
            if random.random() < 0.5:
                img, od, cup = img[::-1, :].copy(), od[::-1, :].copy(), cup[::-1, :].copy()
            k = random.choice([0, 1, 2, 3])
            if k:
                img, od, cup = np.rot90(img, k).copy(), np.rot90(od, k).copy(), np.rot90(cup, k).copy()

        img_t = torch.from_numpy(img.transpose(2, 0, 1))
        mask_t = torch.from_numpy(np.stack([od, cup]).astype(np.float32))
        return img_t, mask_t, r.case, int(r.label), float(r.cdr_expert)


# --------------------------------------------------------------------------- #
# U-Net (identique à drishti_segmentation.py)
# --------------------------------------------------------------------------- #
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, out_channels: int = 2, base: int = 32):
        super().__init__()
        self.e1, self.e2 = conv_block(3, base), conv_block(base, base * 2)
        self.e3, self.e4 = conv_block(base * 2, base * 4), conv_block(base * 4, base * 8)
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
        return self.out(d1)


def dice_loss(probs, target, eps=1e-6):
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def combined_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(torch.sigmoid(logits), target)


def vertical_diameter(mask: np.ndarray) -> float:
    rows = np.where(mask.any(axis=1))[0]
    return float(rows.max() - rows.min() + 1) if len(rows) else 0.0


def compute_cdr(od_mask: np.ndarray, cup_mask: np.ndarray) -> float:
    od_d = vertical_diameter(od_mask)
    return float(vertical_diameter(cup_mask) / od_d) if od_d > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Entraînement du U-Net (segmentation uniquement — le label glaucome/normal
# n'intervient pas ici, seulement les masques OD/cupule)
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = combined_loss(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n


@torch.no_grad()
def val_dice(model, loader, device):
    model.eval()
    dices, n = np.zeros(2), 0
    for x, y, *_ in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        dims = (0, 2, 3)
        inter = (pred * y).sum(dims)
        union = pred.sum(dims) + y.sum(dims)
        dices += ((2 * inter + 1e-6) / (union + 1e-6)).cpu().numpy() * x.size(0)
        n += x.size(0)
    return dices / n


# --------------------------------------------------------------------------- #
# Calcul du CDR pour tous les cas d'un DataLoader, avec le modèle final
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_all_cdr(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, cases, labels, cdr_expert in loader:
        x = x.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()
        for i in range(len(cases)):
            rows.append({
                "case": cases[i],
                "label": int(labels[i]),
                "cdr_expert": float(cdr_expert[i]),
                "cdr_pred": compute_cdr(pred[i, 0], pred[i, 1]),
            })
    return pd.DataFrame(rows)


def youden_threshold(y_true, score):
    fpr, tpr, thr = roc_curve(y_true, score)
    return float(thr[np.argmax(tpr - fpr)])


def diagnosis_metrics(y_true, score, threshold) -> dict:
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "n": int(len(y_true)), "auc_cdr": auc,
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "threshold_cdr": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def save_examples(model, df: pd.DataFrame, size: int, device, out_dir: Path, threshold: float, n: int = 8):
    out = out_dir / "examples"
    out.mkdir(parents=True, exist_ok=True)
    ds = SegDataset(df.sample(min(n, len(df)), random_state=0), size, augment=False)
    for i in range(len(ds)):
        x, y, case, label, cdr_expert = ds[i]
        with torch.no_grad():
            pred = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0].cpu().numpy()
        od_p, cup_p = pred[0] > 0.5, pred[1] > 0.5
        cdr_pred = compute_cdr(od_p, cup_p)
        diag = "GLAUCOME suspecté" if cdr_pred >= threshold else "normal"
        vrai = "glaucome" if label else "normal"

        img = x.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img)
        ax.contour(od_p, colors="lime", linewidths=1.5)
        ax.contour(cup_p, colors="red", linewidths=1.5)
        correct = "✓" if diag.startswith("normal") == (vrai == "normal") else "✗"
        ax.set_title(f"{case} | vrai={vrai} | CDR={cdr_pred:.2f} (seuil={threshold:.2f})\n"
                    f"diagnostic={diag}  {correct}", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{case}.png", dpi=120)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Diagnostic glaucome via CDR (segmentation U-Net) sur Drishti-GS")
    ap.add_argument("--root", required=True, help='ex. "data/drishti gs"')
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs/drishti_cdr")
    ap.add_argument("--fixed_threshold", type=float, default=None,
                    help="Seuil clinique fixe à tester en plus du seuil optimisé, ex. 0.5 ou 0.6")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    df = load_cases(args.root)
    trainval, test = train_test_split(df, test_size=0.2, stratify=df.label, random_state=args.seed)
    train, val = train_test_split(trainval, test_size=0.2, stratify=trainval.label, random_state=args.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s}: {len(part):2d} cas | glaucome={int(part.label.sum())}")

    train_loader = DataLoader(SegDataset(train, args.img_size, True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SegDataset(val, args.img_size, False), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SegDataset(test, args.img_size, False), batch_size=args.batch_size, shuffle=False)

    # --- 1) Entraîner le U-Net à segmenter (le label glaucome/normal n'intervient pas ici) ---
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ckpt = out_dir / "best_unet.pt"
    best_dice, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, device, optimizer)
        scheduler.step()
        vd = val_dice(model, val_loader, device)
        mean_vd = vd.mean()
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={tr_loss:.4f} | "
              f"val Dice OD={vd[0]:.3f} cup={vd[1]:.3f}")
        if mean_vd > best_dice:
            best_dice, best_epoch = mean_vd, epoch
            torch.save(model.state_dict(), ckpt)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping (meilleure epoch : {best_epoch}, Dice moyen={best_dice:.3f})")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # --- 2) Calculer le CDR prédit pour train+val (choix du seuil) et test (évaluation) ---
    trainval_loader = DataLoader(SegDataset(trainval, args.img_size, False), batch_size=args.batch_size, shuffle=False)
    cdr_trainval = compute_all_cdr(model, trainval_loader, device)
    cdr_test = compute_all_cdr(model, test_loader, device)

    # --- 3) Choisir le seuil de CDR sur train+val (jamais sur le test) ---
    thr = youden_threshold(cdr_trainval.label, cdr_trainval.cdr_pred)
    print(f"\nSeuil de CDR retenu (Youden, sur train+val) : {thr:.3f}")

    results = {"threshold_source": "youden_trainval",
              "optimized": diagnosis_metrics(cdr_test.label, cdr_test.cdr_pred, thr)}
    print("\n=== Diagnostic sur le test, à partir du CDR prédit ===")
    print(f"[seuil optimisé={thr:.3f}] " + " | ".join(f"{k}={v}" for k, v in results["optimized"].items()))

    if args.fixed_threshold is not None:
        results["fixed"] = diagnosis_metrics(cdr_test.label, cdr_test.cdr_pred, args.fixed_threshold)
        print(f"[seuil fixe={args.fixed_threshold:.2f}] " + " | ".join(f"{k}={v}" for k, v in results["fixed"].items()))

    # Comparaison avec le CDR des experts, à titre indicatif (mêmes cas de test)
    valid = cdr_test.dropna(subset=["cdr_expert"])
    if not valid.empty:
        mae = float((valid.cdr_pred - valid.cdr_expert).abs().mean())
        results["cdr_mae_vs_expert_on_test"] = mae
        print(f"Écart moyen CDR prédit vs experts (test) : {mae:.3f}")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    cdr_test.to_csv(out_dir / "cdr_test_predictions.csv", index=False)
    save_examples(model, test, args.img_size, device, out_dir, threshold=thr)

    # Courbe ROC (CDR comme score continu)
    if len(np.unique(cdr_test.label)) > 1:
        fpr, tpr, _ = roc_curve(cdr_test.label, cdr_test.cdr_pred)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"CDR (AUC={results['optimized']['auc_cdr']:.3f})")
        plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
        plt.xlabel("1 - Spécificité"); plt.ylabel("Sensibilité")
        plt.title("ROC — diagnostic basé sur le CDR (Drishti-GS)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "roc_cdr.png", dpi=150)
        plt.close()

    print(f"\nRésultats enregistrés dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
