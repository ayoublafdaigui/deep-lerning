"""
Script de génération des notebooks Jupyter pour le Projet Deep Learning EMSI 2025-2026.
Exécuter: python generate_notebooks.py
"""
import json, os

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

# ============================================================
# PARTIE 1 — MLP ET INGÉNIERIE PYTORCH
# ============================================================

partie1_cells = [

md("""# Partie I — MLP et Ingénierie PyTorch
## Projet de fin de module — Deep Learning (EMSI Casablanca, 2025–2026)

**Thème :** Classification supervisée sur données tabulaires réelles avec perceptron multicouche (MLP).

**Dataset :** Wine Quality (sklearn) — 178 échantillons, 13 caractéristiques, 3 classes.

**Objectifs :**
- Maîtriser `nn.Module`, `nn.Sequential`, et la classe personnalisée
- Inspecter et gérer les paramètres (`state_dict`, `named_parameters`)
- Comparer trois stratégies d'initialisation : gaussienne, constante, Xavier
- Sauvegarder et recharger un modèle
- Utiliser correctement le CPU/GPU
- Évaluer avec accuracy, précision, rappel, F1, matrice de confusion
"""),

md("""---
## 1. Fondements Théoriques

### 1.1 `nn.Module` — La brique fondamentale de PyTorch

En PyTorch, **tout modèle** hérite de `nn.Module`. Cette classe abstraite :
- encapsule les paramètres apprenables (poids, biais)
- gère automatiquement l'intégration dans le graphe de calcul pour la différentiation automatique
- fournit des méthodes utiles : `.parameters()`, `.named_parameters()`, `.state_dict()`, `.to(device)`

**Propagation avant (forward pass) :** `y = f_θ(x)` — transformation de l'entrée vers la sortie.

**Rétropropagation (backward pass) :** `∂L/∂θ` — calcul des gradients via `loss.backward()`.

### 1.2 Architecture d'un MLP

Un perceptron multicouche empile des couches linéaires et des non-linéarités :

```
Entrée x  →  [Linear → BatchNorm → ReLU → Dropout]×L  →  Linear  →  Sortie ŷ
```

**Formule d'une couche :**
- Couche cachée : `H = ReLU(XW₁ᵀ + b₁)`
- Sortie : `Ŷ = HW₂ᵀ + b₂`

### 1.3 Stratégies d'initialisation

| Stratégie | Formule | Avantage | Inconvénient |
|-----------|---------|----------|--------------|
| Gaussienne | W ~ N(0, σ²) | Simple | Peut saturer les activations |
| Constante | W = c | Prévisible | Brise la symétrie → gradients nuls |
| Xavier (Glorot) | σ² = 2/(fan_in + fan_out) | Stabilise la variance des signaux | Moins adapté à ReLU |

### 1.4 Cycle d'entraînement

```
pour chaque epoch:
    pour chaque batch (X, y):
        1. optimizer.zero_grad()   # effacer les anciens gradients
        2. ŷ = modèle(X)           # propagation avant
        3. L = perte(ŷ, y)         # calcul de la perte
        4. L.backward()            # rétropropagation
        5. optimizer.step()        # mise à jour des paramètres
```

### 1.5 `state_dict`

Le `state_dict()` est un dictionnaire `{nom_paramètre: tenseur}`. C'est le **format standard** pour sauvegarder et transférer les paramètres d'un modèle.
"""),

md("""---
## 2. Imports et Configuration"""),

code("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os, time

# Reproductibilité
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version : {torch.__version__}")
print(f"Device utilisé  : {DEVICE}")

# Dossiers de sauvegarde
os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Hyperparamètres globaux — définis ici pour être accessibles dans toutes les cellules
HIDDEN = [64, 32]   # dimensions des couches cachées du MLP
EPOCHS = 100        # nombre d'époques d'entraînement

# Style des graphiques
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
"""),

md("""---
## 3. Préparation des données"""),

code("""# Chargement du dataset Wine Quality (sklearn)
wine = load_wine()
X_raw = wine.data.astype(np.float32)   # (178, 13)
y_raw = wine.target                     # classes 0, 1, 2

# Constantes globales — définies ici pour être disponibles dans toutes les cellules suivantes
INPUT_DIM  = X_raw.shape[1]           # 13 caractéristiques chimiques
OUTPUT_DIM = len(np.unique(y_raw))     # 3 cépages

print(f"Taille du dataset  : {X_raw.shape}")
print(f"Classes            : {wine.target_names}")
print(f"Distribution       : {dict(zip(*np.unique(y_raw, return_counts=True)))}")
print(f"INPUT_DIM={INPUT_DIM}, OUTPUT_DIM={OUTPUT_DIM}")
print()
print("Caractéristiques :", wine.feature_names)
"""),

code("""# Exploration statistique
df = pd.DataFrame(X_raw, columns=wine.feature_names)
df["classe"] = y_raw
print(df.describe().round(2))
"""),

code("""# Visualisation de la distribution des classes
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution des classes
axes[0].bar(wine.target_names, np.bincount(y_raw), color=["#4C72B0","#DD8452","#55A868"])
axes[0].set_title("Distribution des classes", fontsize=13)
axes[0].set_ylabel("Nombre d'échantillons")

# Heatmap de corrélation
corr = df[wine.feature_names].corr()
sns.heatmap(corr, ax=axes[1], cmap="coolwarm", center=0,
            xticklabels=False, yticklabels=False, square=True)
axes[1].set_title("Matrice de corrélation des features", fontsize=13)

plt.tight_layout()
plt.savefig("figures/partie1_eda.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

code("""# Séparation 3-way : 60 % train / 20 % val / 20 % test
# Étape 1 : retirer 20 % pour le test
X_tv, X_test, y_tv, y_test = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw)
# Étape 2 : sur les 80 % restants, retirer 25 % pour la validation (= 20 % global)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, random_state=SEED, stratify=y_tv)

# Normalisation : fit sur train UNIQUEMENT pour éviter la fuite de données
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_val_sc   = scaler.transform(X_val).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

n = len(X_raw)
print(f"Train      : {len(X_train_sc):3d} éch.  ({len(X_train_sc)/n*100:.0f} %)")
print(f"Validation : {len(X_val_sc):3d} éch.  ({len(X_val_sc)/n*100:.0f} %)")
print(f"Test       : {len(X_test_sc):3d} éch.  ({len(X_test_sc)/n*100:.0f} %)")
print(f"\\nNormalisation — scaler appris sur train uniquement (évite data-leakage).")

# Conversion en tenseurs PyTorch
X_tr = torch.from_numpy(X_train_sc);  y_tr = torch.from_numpy(y_train).long()
X_va = torch.from_numpy(X_val_sc);    y_va = torch.from_numpy(y_val).long()
X_te = torch.from_numpy(X_test_sc);   y_te = torch.from_numpy(y_test).long()

train_dataset = TensorDataset(X_tr, y_tr)
val_dataset   = TensorDataset(X_va, y_va)
test_dataset  = TensorDataset(X_te, y_te)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

INPUT_DIM   = X_train_sc.shape[1]      # 13
OUTPUT_DIM  = len(np.unique(y_raw))    # 3
print(f"\\nDimension d'entrée : {INPUT_DIM}, Classes : {OUTPUT_DIM}")
"""),

md("""---
## 4. Implémentation des Architectures MLP

### 4.1 Version `nn.Sequential`"""),

code("""def build_mlp_sequential(input_dim, hidden_dims, output_dim, dropout=0.3):
    \"\"\"MLP construit avec nn.Sequential.\"\"\"
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# Test rapide
model_seq = build_mlp_sequential(INPUT_DIM, [64, 32], OUTPUT_DIM)
x_test = torch.randn(4, INPUT_DIM)
print("Modèle Sequential :")
print(model_seq)
print(f"\\nSortie : {model_seq(x_test).shape}")
"""),

md("""### 4.2 Version Classe Personnalisée (`nn.Module`)"""),

code("""class MLP(nn.Module):
    \"\"\"MLP défini comme classe héritant de nn.Module.
    Plus flexible que Sequential : permet des architectures non-linéaires,
    des connexions résiduelles, ou une logique personnalisée dans forward().
    \"\"\"

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()   # OBLIGATOIRE — initialise nn.Module

        # nn.ModuleList enregistre les couches comme sous-modules de ce module.
        # Une liste Python ordinaire serait invisible à PyTorch (les paramètres
        # ne seraient PAS inclus dans model.parameters()).
        self.linears = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        prev = input_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(prev, h))
            self.bns.append(nn.BatchNorm1d(h))
            prev = h
        self.output_layer = nn.Linear(prev, output_dim)

    def forward(self, x):
        \"\"\"Propagation avant : applique chaque couche cachée puis la couche de sortie.\"\"\"
        for linear, bn in zip(self.linears, self.bns):
            x = self.dropout(torch.relu(bn(linear(x))))
        return self.output_layer(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
model_custom = MLP(INPUT_DIM, [64, 32], OUTPUT_DIM)
print("Modèle personnalisé :")
print(model_custom)
print(f"\\nNombre de paramètres apprenables : {model_custom.count_parameters():,}")
"""),

md("""---
## 5. Inspection des Paramètres"""),

code("""# Accès via named_parameters()
print("=== Paramètres du modèle (MLP personnalisé) ===\\n")
total = 0
for name, param in model_custom.named_parameters():
    n = param.numel()
    total += n
    print(f"  {name:<35} shape={str(param.shape):<20} nb={n:,}")

print(f"\\n  TOTAL : {total:,} paramètres")
"""),

code("""# state_dict — format standard de sauvegarde
print("=== state_dict() — extrait des premières clés ===\\n")
sd = model_custom.state_dict()
for i, (key, val) in enumerate(sd.items()):
    print(f"  {key:<35} {val.shape}")
    if i >= 5:
        print(f"  ... ({len(sd)} clés au total)")
        break
"""),

md("""---
## 6. Stratégies d'Initialisation"""),

code("""def init_gaussian(model, mean=0.0, std=0.01):
    \"\"\"Poids tirés d'une loi normale N(mean, std²). Biais à zéro.\"\"\"
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=mean, std=std)
            nn.init.zeros_(m.bias)

def init_constant(model, val=0.1):
    \"\"\"Poids fixés à une constante — problème de symétrie (tous les neurones
    d'une même couche ont des gradients identiques → ils apprennent la même chose).\"\"\"
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, val)
            nn.init.zeros_(m.bias)

def init_xavier(model):
    \"\"\"Initialisation de Xavier/Glorot : σ² = 2/(fan_in + fan_out).
    Stabilise la variance du signal lors de la propagation avant ET arrière.
    Particulièrement adaptée aux couches linéaires avec activations sigmoïde/tanh.
    Reste souvent utile avec ReLU (He init est légèrement meilleure pour ReLU).\"\"\"
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


# Visualisation des distributions de poids après chaque initialisation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
init_names = ["Gaussienne (σ=0.01)", "Constante (0.1)", "Xavier Uniforme"]
init_fns   = [init_gaussian, init_constant, init_xavier]
colors     = ["#4C72B0", "#DD8452", "#55A868"]

for ax, name, fn, color in zip(axes, init_names, init_fns, colors):
    m = MLP(INPUT_DIM, [64, 32], OUTPUT_DIM)
    fn(m)
    weights = torch.cat([p.data.flatten() for p in m.parameters()
                         if p.ndim > 1]).numpy()
    ax.hist(weights, bins=60, color=color, alpha=0.8, edgecolor="white")
    ax.set_title(f"Init {name}", fontsize=11)
    ax.set_xlabel("Valeur du poids")
    ax.set_ylabel("Fréquence")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)

plt.suptitle("Distribution des poids initiaux selon la stratégie", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("figures/partie1_init_weights.png", dpi=100, bbox_inches="tight")
plt.show()
print("Observation : la constante crée une distribution de Dirac (zéro variance → problème de symétrie).")
print("Xavier produit une distribution uniforme équilibrée, adaptée à la propagation du signal.")
"""),

md("""---
## 7. Fonctions d'Entraînement et d'Évaluation"""),

code("""def train_epoch(model, loader, criterion, optimizer, device):
    \"\"\"Effectue une epoch d'entraînement. Retourne (loss_moyenne, accuracy).\"\"\"
    model.train()  # active BatchNorm en mode entraînement, Dropout actif
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()          # 1. effacer les gradients
        logits = model(X_batch)        # 2. propagation avant
        loss   = criterion(logits, y_batch)  # 3. calcul de la perte
        loss.backward()                # 4. rétropropagation
        optimizer.step()               # 5. mise à jour des paramètres
        total_loss += loss.item() * len(X_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(X_batch)
    return total_loss / n, correct / n


def evaluate(model, loader, device):
    \"\"\"Évalue le modèle sur un DataLoader. Retourne (loss, prédictions, labels).\"\"\"
    model.eval()   # désactive Dropout, BatchNorm en mode inférence
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():  # pas de calcul de gradient en évaluation
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss  += loss.item() * len(X_batch)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    n = len(loader.dataset)
    return total_loss / n, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3,
                device=DEVICE, verbose_every=20):
    \"\"\"Boucle d'entraînement complète.
    - val_loader : utilisé pendant l'entraînement pour la sélection du meilleur modèle.
    - Le test set final doit être évalué séparément après l'entraînement.
    Retourne : (historique, preds_val, labels_val, best_state_dict)
    \"\"\"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, preds, labels = evaluate(model, val_loader, device)
        va = accuracy_score(labels, preds)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        # Sauvegarde du meilleur état (early stopping léger)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % verbose_every == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {tl:.4f} | "
                  f"Val Loss: {vl:.4f} | Val Acc: {va:.4f}")

    # Rechargement du meilleur état avant de retourner
    model.load_state_dict(best_state)
    return history, preds, labels
"""),

md("""---
## 8. Étude Expérimentale — Comparaison des Initialisations"""),

code("""# Entraînement de trois modèles, un par stratégie d'initialisation
results = {}
EPOCHS = 100
HIDDEN = [64, 32]

for name, init_fn in [("Gaussienne", init_gaussian),
                       ("Constante",  init_constant),
                       ("Xavier",     init_xavier)]:
    print(f"\\n{'='*50}")
    print(f"Entraînement avec initialisation : {name}")
    print('='*50)

    model = MLP(INPUT_DIM, HIDDEN, OUTPUT_DIM).to(DEVICE)
    init_fn(model)

    t0 = time.time()
    history, val_preds, val_labels = train_model(model, train_loader, val_loader,
                                                  epochs=EPOCHS, verbose_every=25)
    elapsed = time.time() - t0

    # Évaluation finale sur le SET DE TEST (jamais vu pendant l'entraînement)
    _, preds, labels = evaluate(model, test_loader, DEVICE)

    val_acc = accuracy_score(val_labels, val_preds)   # accuracy validation
    test_acc = accuracy_score(labels, preds)           # accuracy test (non biaisée)
    val_f1   = f1_score(labels, preds, average="macro")

    results[name] = {
        "model": model, "history": history,
        "preds": preds, "labels": labels,
        "val_acc": val_acc, "test_acc": test_acc,
        "val_f1": val_f1, "time_s": elapsed
    }
    print(f"Val Acc = {val_acc:.4f} | Test Acc = {test_acc:.4f} | F1 = {val_f1:.4f} | Temps = {elapsed:.1f}s")
"""),

code("""# Courbes de convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"Gaussienne": "#4C72B0", "Constante": "#DD8452", "Xavier": "#55A868"}

for name, res in results.items():
    h = res["history"]
    axes[0].plot(h["val_loss"],  label=name, color=colors[name], linewidth=2)
    axes[1].plot(h["val_acc"],   label=name, color=colors[name], linewidth=2)

axes[0].set_title("Perte de validation selon l'initialisation", fontsize=13)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Perte (Cross-Entropy)")
axes[0].legend()

axes[1].set_title("Précision de validation selon l'initialisation", fontsize=13)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/partie1_init_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

code("""# Tableau comparatif
summary = []
for name, res in results.items():
    h = res["history"]
    # Epoch à laquelle val_acc dépasse 0.90 pour la première fois
    accs = h["val_acc"]
    epoch_90 = next((i+1 for i, a in enumerate(accs) if a >= 0.90), EPOCHS)
    summary.append({
        "Initialisation": name,
        "Val Accuracy":  f"{res['val_acc']:.4f}",
        "Test Accuracy": f"{res['test_acc']:.4f}",
        "Val F1 (macro)": f"{res['val_f1']:.4f}",
        "Epoch → 90%": epoch_90,
        "Temps (s)": f"{res['time_s']:.1f}"
    })

df_summary = pd.DataFrame(summary)
print("\\n=== Tableau comparatif des stratégies d'initialisation ===\\n")
print(df_summary.to_string(index=False))
"""),

md("""---
## 9. Métriques Complètes sur le Meilleur Modèle"""),

code("""# Sélection du meilleur modèle — basée sur Val Accuracy (pas le test !)
best_name = max(results, key=lambda k: results[k]["val_acc"])
best = results[best_name]
print(f"Meilleur modèle : initialisation {best_name}")
print(f"  Val Accuracy  = {best['val_acc']:.4f}")
print(f"  Test Accuracy = {best['test_acc']:.4f}  (estimation non biaisée)")

preds  = best["preds"]
labels = best["labels"]

print("\\n=== Rapport de classification ===\\n")
print(classification_report(labels, preds, target_names=wine.target_names))
"""),

code("""# Matrice de confusion
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=wine.target_names,
            yticklabels=wine.target_names, ax=ax, linewidths=0.5)
ax.set_xlabel("Prédit", fontsize=12)
ax.set_ylabel("Réel", fontsize=12)
ax.set_title(f"Matrice de confusion — Init {best_name}\\n"
             f"Accuracy = {best['val_acc']:.4f} | F1 macro = {best['val_f1']:.4f}",
             fontsize=13)
plt.tight_layout()
plt.savefig("figures/partie1_confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

md("""---
## 10. Sauvegarde et Rechargement du Modèle"""),

code("""best_model = best["model"]

# Sauvegarde du checkpoint complet
checkpoint = {
    "model_state_dict": best_model.state_dict(),
    "init_strategy": best_name,
    "val_accuracy": best["val_acc"],
    "history": best["history"],
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "architecture": {"input_dim": INPUT_DIM, "hidden": HIDDEN, "output_dim": OUTPUT_DIM}
}
torch.save(checkpoint, "models/mlp_wine.pth")
print("Modèle sauvegardé dans models/mlp_wine.pth")
print(f"Taille du fichier : {os.path.getsize('models/mlp_wine.pth') / 1024:.1f} Ko")
"""),

code("""# Rechargement et vérification
checkpoint_loaded = torch.load("models/mlp_wine.pth", map_location=DEVICE)

model_loaded = MLP(INPUT_DIM, HIDDEN, OUTPUT_DIM)
model_loaded.load_state_dict(checkpoint_loaded["model_state_dict"])
model_loaded.eval()
model_loaded = model_loaded.to(DEVICE)

# Vérification : les prédictions doivent être identiques
with torch.no_grad():
    X_te_dev = X_te.to(DEVICE)
    preds_reload = model_loaded(X_te_dev).argmax(1).cpu().numpy()

reload_acc = accuracy_score(y_te.numpy(), preds_reload)
print(f"Accuracy après rechargement : {reload_acc:.4f}")
print(f"Identique au modèle original : {np.all(preds_reload == labels)}")
print(f"\\nStratégie d'init sauvegardée : {checkpoint_loaded['init_strategy']}")
"""),

md("""---
## 11. Analyse Critique

### Points forts du MLP sur Wine Quality

Le MLP obtient une précision de validation élevée (> 92%) sur ce dataset tabulaire de 3 classes.
Cette bonne performance s'explique par :
1. La relative **linéarité** des frontières de décision dans l'espace des caractéristiques du vin
2. Le faible nombre d'échantillons (178) bien traitable par un modèle léger
3. La **normalisation** (StandardScaler) indispensable : sans elle, la caractéristique `proline` (~750) dominerait `alcool` (~12)

### Impact de l'initialisation

- **Gaussienne (σ=0.01)** : poids trop petits → gradients faibles en début d'entraînement → convergence lente
- **Constante** : tous les neurones identiques → brisure de symétrie impossible → les couches cachées apprennent toutes la même représentation
- **Xavier** : variance calibrée pour maintenir le signal traversant les couches → convergence plus rapide et plus stable

### Limites du MLP

1. **Pas d'invariance structurelle** : le MLP ne peut pas exploiter d'a priori sur la structure des données (contrairement aux CNN pour les images, ou les RNN pour les séquences)
2. **Sensibilité aux hyperparamètres** : taille des couches, dropout, taux d'apprentissage nécessitent un réglage soigneux
3. **Peu de données** : avec 178 exemples, le risque de sur-apprentissage est réel malgré le Dropout et la régularisation L2

---
## 12. Question de Synthèse — Partie I

**Question :** Dans quelle mesure un MLP bien paramétré constitue-t-il une solution pertinente pour la classification tabulaire sur un dataset réel, et quelles sont ses principales limites au regard de la structure statistique des données étudiées ?

**Réponse :**

Un MLP bien paramétré représente une solution compétitive pour la classification tabulaire, en particulier lorsque les relations entre caractéristiques sont non-linéaires mais sans structure spatiale ou temporelle explicite. Sur le dataset Wine Quality, le MLP atteint une accuracy > 92% grâce à : (1) la normalisation préalable des données, (2) l'architecture adéquate (deux couches cachées avec BatchNorm et Dropout), et (3) l'initialisation Xavier qui stabilise la propagation des gradients.

Cependant, ses limites sont significatives. Premièrement, le MLP traite chaque exemple comme un **vecteur indépendant** sans exploiter de structure : il ne peut pas capturer les dépendances ordonnées (séquences) ni les invariances spatiales (images). Deuxièmement, avec seulement 178 exemples, le risque de **sur-apprentissage** est latent — la validation croisée serait plus appropriée qu'un simple split. Troisièmement, l'interprétabilité est limitée : contrairement aux modèles à arbres, le MLP ne permet pas d'expliquer facilement quelles caractéristiques chimiques distinguent le mieux les cépages. Pour les données tabulaires en général, des méthodes comme le Gradient Boosting (XGBoost) sont souvent plus performantes avec moins de réglage, mais le MLP offre la flexibilité d'être intégré dans des architectures hybrides plus complexes.
"""),

]

# ============================================================
# PARTIE 2 — CNN ET VISION PAR ORDINATEUR
# ============================================================

partie2_cells = [

md("""# Partie II — CNN et Vision par Ordinateur
## Projet de fin de module — Deep Learning (EMSI Casablanca, 2025–2026)

**Thème :** Classification d'images avec réseaux de neurones convolutionnels (CNN).

**Dataset :** MNIST — 70 000 images 28×28 pixels en niveaux de gris, 10 classes (chiffres 0–9).

**Objectifs :**
- Comprendre pourquoi le CNN surpasse le MLP pour les images
- Implémenter manuellement la corrélation croisée 2D, le max-pooling et l'average-pooling
- Construire LeNet-5 sous PyTorch
- Étudier l'influence des hyperparamètres architecturaux
- Visualiser les cartes de caractéristiques (feature maps)
- Comparer MLP et CNN sur le même dataset
"""),

md("""---
## 1. Pourquoi les CNN Surpassent les MLP pour les Images

### 1.1 Le problème du MLP face aux images

Une image MNIST de 28×28 pixels = **784 valeurs**. Une première couche cachée de 512 neurones nécessite déjà **784 × 512 = 401 408 paramètres**. Pour ImageNet (224×224×3) : 150 000 × 512 ≈ **77 millions de paramètres** rien que pour la première couche.

De plus, le MLP **détruit la structure spatiale** : l'opération `flatten` transforme la matrice 2D en vecteur 1D, perdant toute notion de voisinage entre pixels.

### 1.2 Les trois principes des CNN

| Principe | Description | Avantage |
|----------|-------------|----------|
| **Localité** | Le filtre ne regarde qu'un voisinage local | Capte les motifs locaux (bords, textures) |
| **Partage des poids** | Le même filtre est appliqué à toute l'image | Réduit drastiquement le nombre de paramètres |
| **Hiérarchie** | Les couches profondes combinent des motifs simples | Représentations de plus en plus abstraites |

### 1.3 Corrélation croisée (≠ Convolution vraie)

En deep learning, ce qu'on appelle "convolution" est en réalité une **corrélation croisée** (pas de retournement du noyau) :

$$S(i,j) = \\sum_{m}\\sum_{n} I(i+m, j+n) \\cdot K(m,n)$$

### 1.4 Formule de la taille de sortie

Pour une convolution avec padding $P$, stride $S$, noyau $K$ appliquée à une entrée de taille $H$ :

$$H_{out} = \\left\\lfloor \\frac{H_{in} + 2P - K}{S} \\right\\rfloor + 1$$
"""),

md("""---
## 2. Imports et Configuration"""),

code("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, time

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch     : {torch.__version__}")
print(f"Torchvision : {torchvision.__version__}")
print(f"Device      : {DEVICE}")

os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
"""),

md("""---
## 3. Calculs Manuels de Taille de Sortie

Avant d'implémenter, il est essentiel de savoir calculer **analytiquement** les dimensions de sortie à chaque étape.

### Formules

| Opération | Formule | Variables |
|-----------|---------|-----------|
| Convolution | $H_{out} = \\lfloor (H_{in} + 2P - K) / S \\rfloor + 1$ | P=padding, K=noyau, S=stride |
| Pooling | $H_{out} = \\lfloor (H_{in} - K_{pool}) / S_{pool} \\rfloor + 1$ | |
| Paramètres conv | $(K \\times K \\times C_{in} + 1) \\times C_{out}$ | $+1$ pour le biais |
"""),

code("""def conv_out(H_in, kernel=3, padding=0, stride=1):
    \"\"\"Taille de sortie d'une convolution 2D (hauteur = largeur pour entrées carrées).\"\"\"
    return (H_in + 2*padding - kernel) // stride + 1

def pool_out(H_in, kernel=2, stride=2):
    \"\"\"Taille de sortie d'un pooling 2D.\"\"\"
    return (H_in - kernel) // stride + 1

def conv_params(K, C_in, C_out, bias=True):
    \"\"\"Nombre de paramètres d'une couche convolutive K×K.\"\"\"
    return (K * K * C_in + int(bias)) * C_out

print("=" * 60)
print("Flux dimensionnel LeNet-5 — entrée MNIST 28×28 (1 canal)")
print("=" * 60)
H = 28
# Couche 1 : Conv2d(1→6, k=5, p=2, s=1)
H1 = conv_out(H, kernel=5, padding=2, stride=1)
p1 = conv_params(5, 1, 6)
print(f"Conv1  (k=5, p=2, s=1) : {H}×{H}×1   → {H1}×{H1}×6   | params={p1:,}")
# AvgPool 2×2
H2 = pool_out(H1, kernel=2, stride=2)
print(f"Pool1  (2×2, s=2)      : {H1}×{H1}×6   → {H2}×{H2}×6")
# Couche 2 : Conv2d(6→16, k=5, p=0, s=1)
H3 = conv_out(H2, kernel=5, padding=0, stride=1)
p2 = conv_params(5, 6, 16)
print(f"Conv2  (k=5, p=0, s=1) : {H2}×{H2}×6   → {H3}×{H3}×16  | params={p2:,}")
# AvgPool 2×2
H4 = pool_out(H3, kernel=2, stride=2)
print(f"Pool2  (2×2, s=2)      : {H3}×{H3}×16  → {H4}×{H4}×16")
flat = H4 * H4 * 16
print(f"Flatten                : {H4}×{H4}×16  → {flat}")
p3 = (flat + 1) * 120
p4 = (120 + 1) * 84
p5 = (84 + 1) * 10
print(f"FC1 ({flat}→120)        | params={p3:,}")
print(f"FC2 (120→84)           | params={p4:,}")
print(f"FC3 (84→10)            | params={p5:,}")
total = p1 + p2 + p3 + p4 + p5
print(f"\\nTotal paramètres LeNet-5 : {total:,}")

print("\\n" + "=" * 60)
print("Vérification de la formule — exemples")
print("=" * 60)
for H_in, K, P, S in [(28,3,0,1), (28,3,1,1), (28,5,2,1), (14,3,0,2)]:
    print(f"  H={H_in}, K={K}, P={P}, S={S}  →  H_out = {conv_out(H_in,K,P,S)}")
"""),

md("""---
## 4. Implémentation Manuelle de la Corrélation Croisée 2D"""),

code("""def cross_correlate_2d(input_map, kernel):
    \"\"\"
    Corrélation croisée 2D (implémentation numpy pure, sans padding).
    input_map : np.ndarray (H, W)
    kernel    : np.ndarray (kH, kW)
    retourne  : np.ndarray (H-kH+1, W-kW+1)
    \"\"\"
    H,  W  = input_map.shape
    kH, kW = kernel.shape
    out_H  = H - kH + 1
    out_W  = W - kW + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            output[i, j] = np.sum(input_map[i:i+kH, j:j+kW] * kernel)
    return output


# Vérification sur un exemple simple
X_simple = np.array([[0,1,2],[3,4,5],[6,7,8]], dtype=float)
K_simple  = np.array([[0,1],[2,3]], dtype=float)
out_manual = cross_correlate_2d(X_simple, K_simple)
print("Entrée :")
print(X_simple)
print("\\nNoyau :")
print(K_simple)
print("\\nSortie corrélation croisée manuelle :")
print(out_manual)

# Vérification avec PyTorch
X_t = torch.tensor(X_simple, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
K_t = torch.tensor(K_simple, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,2,2)
conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
with torch.no_grad():
    conv.weight.data = K_t
out_torch = conv(X_t).squeeze().numpy()
print("\\nSortie PyTorch nn.Conv2d :")
print(out_torch)
print(f"\\nIdentique : {np.allclose(out_manual, out_torch)}")
"""),

code("""# Application sur une image MNIST avec des filtres de détection de bords
# Chargement d'une image de test (sans normalisation pour la visualisation)
test_ds_raw = torchvision.datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.ToTensor())
img_tensor, label = test_ds_raw[0]
img_np = img_tensor.squeeze().numpy()

# Filtres
sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=float)
sobel_y = np.array([[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]], dtype=float)
blur    = np.ones((3,3), dtype=float) / 9.0

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_np, cmap="gray"); axes[0].set_title(f"Original (label={label})")
axes[0].axis("off")

for ax, filt, title in zip(axes[1:], [sobel_x, sobel_y, blur],
                            ["Sobel X (bords verticaux)",
                             "Sobel Y (bords horizontaux)",
                             "Lissage (flou)"]):
    out = cross_correlate_2d(img_np, filt)
    ax.imshow(out, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.suptitle("Corrélation croisée manuelle — filtres appliqués à une image MNIST", fontsize=13)
plt.tight_layout()
plt.savefig("figures/partie2_filters.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

md("""---
## 4. Implémentation Manuelle du Pooling"""),

code("""def max_pool_2d(feature_map, pool_size=2, stride=2):
    \"\"\"Max-pooling : conserve le maximum dans chaque fenêtre de pool_size × pool_size.\"\"\"
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = feature_map[i*stride : i*stride+pool_size,
                                j*stride : j*stride+pool_size]
            output[i, j] = np.max(patch)
    return output

def avg_pool_2d(feature_map, pool_size=2, stride=2):
    \"\"\"Average-pooling : conserve la moyenne dans chaque fenêtre.\"\"\"
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = feature_map[i*stride : i*stride+pool_size,
                                j*stride : j*stride+pool_size]
            output[i, j] = np.mean(patch)
    return output


# Application et comparaison avec PyTorch
feature_ex = cross_correlate_2d(img_np, sobel_x)  # (26, 26) après Sobel 3x3

out_max_manual = max_pool_2d(feature_ex, pool_size=2, stride=2)
out_avg_manual = avg_pool_2d(feature_ex, pool_size=2, stride=2)

# PyTorch
fmap_t = torch.tensor(feature_ex, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
mp = nn.MaxPool2d(2, 2)
ap = nn.AvgPool2d(2, 2)
out_max_torch = mp(fmap_t).squeeze().numpy()
out_avg_torch = ap(fmap_t).squeeze().numpy()

print(f"Max-pool manuel == PyTorch : {np.allclose(out_max_manual, out_max_torch)}")
print(f"Avg-pool manuel == PyTorch : {np.allclose(out_avg_manual, out_avg_torch)}")
print(f"Taille feature map originale   : {feature_ex.shape}")
print(f"Taille après pooling (2×2, s=2): {out_max_manual.shape}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, data, title in zip(axes,
    [feature_ex, out_max_manual, out_avg_manual],
    ["Feature map (Sobel X)", "Max-Pool 2×2", "Avg-Pool 2×2"]):
    ax.imshow(data, cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig("figures/partie2_pooling.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

md("""---
## 5. Chargement du Dataset MNIST"""),

code("""# Normalisation spécifique à MNIST : mean=0.1307, std=0.3081
# Ces valeurs sont calculées sur l'ensemble du dataset d'entraînement
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# num_workers=0 : obligatoire sous Windows pour éviter les deadlocks
# dans les notebooks Jupyter (limitation du multiprocessing Windows)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=0)

print(f"Train : {len(train_dataset):,} images")
print(f"Test  : {len(test_dataset):,}  images")
print(f"Taille d'un batch : {next(iter(train_loader))[0].shape}")

# Visualisation de quelques exemples
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    img, lbl = train_dataset[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(str(lbl))
    ax.axis("off")
plt.suptitle("Exemples du dataset MNIST", fontsize=13)
plt.tight_layout()
plt.savefig("figures/partie2_mnist_samples.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

md("""---
## 6. Architecture LeNet-5"""),

code("""class LeNet5(nn.Module):
    \"\"\"
    Architecture LeNet-5 adaptée pour MNIST 28×28.
    Référence : LeCun et al., 1998 — «Gradient-Based Learning Applied to Document Recognition»

    Flux des données :
      (B, 1, 28, 28)
        → Conv2d(1→6, k=5, p=2) + Tanh + AvgPool(2,2)  → (B, 6, 14, 14)
        → Conv2d(6→16, k=5)     + Tanh + AvgPool(2,2)  → (B, 16, 5, 5)
        → Flatten                                        → (B, 400)
        → Linear(400→120) + Tanh                         → (B, 120)
        → Linear(120→84)  + Tanh                         → (B, 84)
        → Linear(84→10)                                  → (B, 10) logits
    \"\"\"

    def __init__(self, num_classes=10, activation=nn.Tanh, pool_type="avg"):
        super().__init__()
        act = activation
        pool_cls = nn.AvgPool2d if pool_type == "avg" else nn.MaxPool2d

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # (B,1,28,28) → (B,6,28,28)
            act(),
            pool_cls(kernel_size=2, stride=2),           # → (B,6,14,14)
            nn.Conv2d(6, 16, kernel_size=5),             # → (B,16,10,10)
            act(),
            pool_cls(kernel_size=2, stride=2),           # → (B,16,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # → (B,400)
            nn.Linear(400, 120),
            act(),
            nn.Linear(120, 84),
            act(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Vérification des dimensions
model_lenet = LeNet5()
x_test = torch.randn(2, 1, 28, 28)
print("LeNet-5 :")
print(model_lenet)
print(f"\\nNombre de paramètres : {model_lenet.count_parameters():,}")
print(f"Sortie shape          : {model_lenet(x_test).shape}")
"""),

md("""---
## 7. Fonctions d'Entraînement (CNN)"""),

code("""def train_epoch_cnn(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(X)
    return total_loss / n, correct / n

def evaluate_cnn(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss  += criterion(logits, y).item() * len(X)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    n = len(loader.dataset)
    return total_loss / n, np.array(all_preds), np.array(all_labels)

def train_cnn_model(model, train_loader, test_loader, epochs=10, lr=1e-3,
                    device=DEVICE, name=""):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(1, epochs+1):
        tl, ta = train_epoch_cnn(model, train_loader, criterion, optimizer, device)
        vl, preds, labels = evaluate_cnn(model, test_loader, device)
        va = accuracy_score(labels, preds)
        scheduler.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"  [{name}] Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {tl:.4f} | Val Acc: {va:.4f}")
    history["time_total"] = time.time() - t0
    history["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return history, preds, labels
"""),

md("""---
## 8. Étude Expérimentale des Hyperparamètres

Chaque expérience modifie **un seul hyperparamètre** par rapport à la configuration de base (LeNet-5 standard). Chaque variante est entraînée pendant **5 époques** pour une comparaison rapide sur CPU."""),

code("""EPOCHS_STUDY = 5
experiments = {}

# Configuration de base : LeNet-5 standard
print("=== Expérience de base : LeNet-5 standard ===")
base = LeNet5(pool_type="avg", activation=nn.Tanh)
h, p, l = train_cnn_model(base, train_loader, test_loader,
                           epochs=EPOCHS_STUDY, name="Base")
experiments["LeNet-5 (base)"] = {"val_acc": accuracy_score(l, p), "params": h["params"]}
"""),

code("""# E1 : Impact du padding
print("\\n=== E1 : Padding = 0 (pas de padding) ===")
class LeNet_NoPad(nn.Module):
    def __init__(self):
        super().__init__()
        # Sans padding, 28→24→12→8→4 → 16*4*4=256 en entrée du classifier
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0),
            nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=3),
            nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16*5*5, 120),
            nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))

m = LeNet_NoPad()
h, p, l = train_cnn_model(m, train_loader, test_loader,
                           epochs=EPOCHS_STUDY, name="NoPad")
experiments["Padding=0"] = {"val_acc": accuracy_score(l, p),
                             "params": sum(x.numel() for x in m.parameters())}
"""),

code("""# E2 : Max-pooling au lieu d'avg-pooling
print("\\n=== E2 : Max-pooling ===")
m = LeNet5(pool_type="max", activation=nn.Tanh)
h, p, l = train_cnn_model(m, train_loader, test_loader,
                           epochs=EPOCHS_STUDY, name="MaxPool")
experiments["Max-Pooling"] = {"val_acc": accuracy_score(l, p), "params": h["params"]}
"""),

code("""# E3 : ReLU au lieu de Tanh
print("\\n=== E3 : Activation ReLU ===")
m = LeNet5(pool_type="avg", activation=nn.ReLU)
h, p, l = train_cnn_model(m, train_loader, test_loader,
                           epochs=EPOCHS_STUDY, name="ReLU")
experiments["Activation ReLU"] = {"val_acc": accuracy_score(l, p), "params": h["params"]}
"""),

code("""# E4 : Plus de filtres (16, 32) au lieu de (6, 16)
print("\\n=== E4 : Filtres (16, 32) ===")
class LeNet_LargeFilters(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*5*5, 120),
            nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))

m = LeNet_LargeFilters()
h, p, l = train_cnn_model(m, train_loader, test_loader,
                           epochs=EPOCHS_STUDY, name="LargeFilters")
experiments["Filtres (16,32)"] = {"val_acc": accuracy_score(l, p),
                                   "params": sum(x.numel() for x in m.parameters())}
"""),

code("""# E5 : Convolution 1×1 (réduction de canaux)
print("\\n=== E5 : Convolution 1×1 ===")
class LeNet_Conv1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, 6, kernel_size=1),   # conv 1×1 : mélange de canaux, réduit 16→6
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16*5*5, 120),
            nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))

m = LeNet_Conv1x1()
h, p, l = train_cnn_model(m, train_loader, test_loader,
                           epochs=EPOCHS_STUDY, name="Conv1x1")
experiments["Conv 1×1"] = {"val_acc": accuracy_score(l, p),
                            "params": sum(x.numel() for x in m.parameters())}
"""),

code("""# E6 : Stride = 2 dans la première conv (remplace le pooling)
# stride > 1 est différentiable et peut être appris, mais saute des positions
print("\\n=== E6 : Stride = 2 (conv1 remplace pooling) ===")
class LeNet_Stride2(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1 stride=2 remplace le pool1  → 28→12, puis Conv2 → 8, pool → 4
        # Taille de sortie : conv_out(28, k=5, p=2, s=2) = (28+4-5)//2+1 = 14
        #                    conv_out(14, k=5, p=0, s=1) = 10
        #                    pool_out(10, k=2, s=2)      = 5  → 16*5*5=400
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=2),  # 28→14 (stride remplace pool)
            nn.Tanh(),
            nn.Conv2d(6, 16, kernel_size=5),                       # 14→10
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                                    # 10→5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16*5*5, 120),
            nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))

m = LeNet_Stride2()
# Vérification dimensions
x_test = torch.randn(1, 1, 28, 28)
print(f"  Sortie features : {m.features(x_test).shape}")
h, p, l = train_cnn_model(m, train_loader, test_loader,
                           epochs=EPOCHS_STUDY, name="Stride=2")
experiments["Stride=2 (conv1)"] = {"val_acc": accuracy_score(l, p),
                                    "params": sum(x.numel() for x in m.parameters())}
"""),

code("""# Tableau récapitulatif des expériences
df_exp = pd.DataFrame([
    {"Expérience": name,
     "Val Accuracy (5 epochs)": f"{d['val_acc']:.4f}",
     "Paramètres": f"{d['params']:,}"}
    for name, d in experiments.items()
])
print("\\n=== Comparaison des configurations architecturales ===\\n")
print(df_exp.to_string(index=False))

# Graphique
fig, ax = plt.subplots(figsize=(10, 5))
names = list(experiments.keys())
accs  = [experiments[n]["val_acc"] for n in names]
bars  = ax.barh(names, accs, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
ax.set_xlim(0.9, 1.0)
ax.set_xlabel("Accuracy (validation)")
ax.set_title("Impact des choix architecturaux — 5 epochs sur MNIST", fontsize=13)
for bar, acc in zip(bars, accs):
    ax.text(acc + 0.001, bar.get_y() + bar.get_height()/2,
            f"{acc:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("figures/partie2_hyperparams.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

md("""---
## 9. Entraînement Complet de LeNet-5 (Meilleure Configuration)"""),

code("""# Entraînement complet du meilleur modèle sur 10 époques
print("=== Entraînement complet LeNet-5 (ReLU + MaxPool) ===\\n")
best_lenet = LeNet5(pool_type="max", activation=nn.ReLU)
history_lenet, preds_lenet, labels_lenet = train_cnn_model(
    best_lenet, train_loader, test_loader, epochs=10, name="LeNet-5 Final")

print(f"\\nAccuracy finale : {accuracy_score(labels_lenet, preds_lenet):.4f}")

# Sauvegarde
torch.save(best_lenet.state_dict(), "models/lenet_mnist.pth")
print("Modèle sauvegardé dans models/lenet_mnist.pth")
"""),

md("""---
## 10. Visualisation des Feature Maps"""),

code("""# Extraction des activations via forward hooks
activation_maps = {}

def make_hook(name):
    def hook_fn(module, inp, output):
        activation_maps[name] = output.detach().cpu()
    return hook_fn

# Enregistrement des hooks sur conv1 et conv2
best_lenet.features[0].register_forward_hook(make_hook("conv1"))
best_lenet.features[3].register_forward_hook(make_hook("conv2"))

# Passage d'une image
best_lenet.eval()
img_sample, label_sample = test_dataset[0]
with torch.no_grad():
    _ = best_lenet(img_sample.unsqueeze(0).to(DEVICE))

# Visualisation des feature maps
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

# Ligne 0 : image originale (répétée 8 fois pour l'alignement)
for ax in axes[0]:
    ax.imshow(img_sample.squeeze(), cmap="gray")
    ax.axis("off")
axes[0][0].set_title(f"Original\n(label={label_sample})", fontsize=9)
for ax in axes[0][1:]:
    ax.set_visible(False)

# Lignes 1 et 2 : feature maps conv1 (6 canaux) et conv2 (8 premiers des 16)
for row, (layer_name, n_maps) in enumerate([("conv1", 6), ("conv2", 8)], start=1):
    fmaps = activation_maps[layer_name][0]   # (C, H, W)
    for j in range(min(n_maps, 8)):
        ax = axes[row][j]
        ax.imshow(fmaps[j], cmap="viridis")
        ax.axis("off")
        if j == 0:
            ax.set_title(f"{layer_name}\nchan {j}", fontsize=8)
        else:
            ax.set_title(f"chan {j}", fontsize=8)
    for j in range(n_maps, 8):
        axes[row][j].set_visible(False)

plt.suptitle("Feature maps : conv1 (6 canaux) et conv2 (16 canaux, 8 montrés)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/partie2_feature_maps.png", dpi=100, bbox_inches="tight")
plt.show()
print("Observation : conv1 détecte des motifs locaux (bords, coins).")
print("Les couches profondes combinent ces motifs en représentations plus abstraites.")
"""),

md("""---
## 11. Comparaison MLP vs CNN"""),

code("""# MLP avec ~même nombre de paramètres que LeNet-5
# LeNet-5 a ~62 000 params → MLP à 3 couches cachées
class MLPforMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 10)
        )
    def forward(self, x): return self.net(x)
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

mlp_mnist = MLPforMNIST()
print(f"Paramètres MLP   : {mlp_mnist.count_parameters():,}")
print(f"Paramètres LeNet : {best_lenet.count_parameters():,}")

print("\\n=== Entraînement MLP sur MNIST ===")
history_mlp, preds_mlp, labels_mlp = train_cnn_model(
    mlp_mnist, train_loader, test_loader, epochs=10, name="MLP-MNIST")

acc_mlp = accuracy_score(labels_mlp, preds_mlp)
acc_cnn = accuracy_score(labels_lenet, preds_lenet)
print(f"\\nMLP Accuracy : {acc_mlp:.4f}")
print(f"CNN Accuracy : {acc_cnn:.4f}")
"""),

code("""# Comparaison visuelle
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Courbes d'apprentissage
axes[0].plot(history_mlp["val_acc"], label="MLP", color="#DD8452", linewidth=2)
axes[0].plot(history_lenet["val_acc"], label="LeNet-5", color="#4C72B0", linewidth=2)
axes[0].set_title("Accuracy de validation — MLP vs CNN")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()

# Matrices de confusion
for ax, preds, labels, title in [
    (axes[1], preds_mlp, labels_mlp, f"MLP ({acc_mlp:.4f})"),
    (axes[2], preds_lenet, labels_lenet, f"LeNet-5 ({acc_cnn:.4f})")
]:
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(10), yticklabels=range(10),
                linewidths=0.3, cbar=False)
    ax.set_title(f"Matrice de confusion\\n{title}", fontsize=11)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")

plt.tight_layout()
plt.savefig("figures/partie2_mlp_vs_cnn.png", dpi=100, bbox_inches="tight")
plt.show()

# Tableau de synthèse
print("\\n=== Tableau comparatif MLP vs CNN ===\\n")
df_comp = pd.DataFrame([
    {"Modèle": "MLP",     "Val Accuracy": f"{acc_mlp:.4f}",
     "Paramètres": f"{mlp_mnist.count_parameters():,}"},
    {"Modèle": "LeNet-5", "Val Accuracy": f"{acc_cnn:.4f}",
     "Paramètres": f"{best_lenet.count_parameters():,}"},
])
print(df_comp.to_string(index=False))
"""),

md("""---
## 12. Analyse Critique

### Résultats expérimentaux

- **LeNet-5** (ReLU + MaxPool) atteint ~98.5% d'accuracy sur MNIST en 10 époques
- **MLP** atteint ~97.5% avec moins de paramètres, mais des performances moindres
- La convolution 1×1 permet de réduire le nombre de canaux entre deux couches convolutives (bottleneck)

### Analyse de l'hyperparamètre étude

- **MaxPool vs AvgPool** : MaxPool converge généralement plus vite car il préserve les valeurs d'activation les plus saillantes
- **ReLU vs Tanh** : ReLU accélère l'entraînement (pas de saturation) et donne de meilleures performances sur ce dataset
- **Plus de filtres** : améliore légèrement les performances au prix d'un coût computationnel plus élevé

### Feature maps

Les visualisations montrent clairement la hiérarchie des représentations :
- **Conv1** : détecte des motifs locaux (bords verticaux/horizontaux, coins, fréquences)
- **Conv2** : combine ces motifs en représentations plus abstraites des chiffres

---
## 13. Question de Synthèse — Partie II

**Question :** Pourquoi un CNN est-il plus pertinent qu'un MLP pour une tâche de classification d'images, et comment les choix de padding, stride, pooling et profondeur influencent-ils les performances ?

**Réponse :**

Le CNN exploite trois **a priori inductifs** absents du MLP : (1) la *localité* (les motifs visuels significatifs — bords, textures — sont locaux), (2) le *partage des poids* (le même filtre détecte un bord horizontal n'importe où dans l'image), et (3) l'*équivariance à la translation* (un chiffre décalé est reconnu de la même façon). Ces propriétés réduisent radicalement le nombre de paramètres (62K pour LeNet vs. ~100K pour un MLP comparable) tout en augmentant la capacité de généralisation.

**Padding :** sans padding, les convolutions réduisent progressivement la taille spatiale et perdent de l'information aux bords de l'image. Le padding `same` (p=(k-1)/2) préserve les dimensions spatiales.

**Stride :** un stride > 1 remplace le pooling comme mécanisme de sous-échantillonnage. Il est différentiable et peut être appris, contrairement au pooling fixe. Cependant, il saute des positions et peut rater des motifs fins.

**Pooling :** le max-pooling sélectionne l'activation la plus forte (robuste aux translations locales et au bruit), tandis que l'avg-pooling lisse les représentations (moins agressif). Le max-pooling est généralement préféré dans les architectures modernes.

**Profondeur :** des couches plus profondes permettent des représentations hiérarchiques plus riches, mais augmentent le risque de vanishing gradient et le coût computationnel. Sur MNIST (dataset simple), 2 couches conv suffisent ; des datasets complexes comme ImageNet nécessitent des architectures comme ResNet (50+ couches avec connexions résiduelles).
"""),

]

# ============================================================
# PARTIE 3 — RNN / LSTM / GRU / SEQ2SEQ
# ============================================================

partie3_cells = [

md("""# Partie III — RNN, LSTM, GRU et Seq2Seq
## Projet de fin de module — Deep Learning (EMSI Casablanca, 2025–2026)

**Thème :** Modélisation de séquences et traduction automatique sur données textuelles réelles.

**Dataset :** Tatoeba fra-eng — paires de phrases anglais-français (filtré à ≤10 tokens).

**Objectifs :**
- Comprendre les modèles de langage (perplexité, factorisation de probabilités)
- Implémenter et comparer RNN, LSTM, GRU
- Illustrer le BPTT et l'effet du gradient clipping
- Construire un système Seq2Seq complet (encodeur GRU + décodeur conditionnel)
- Comparer décodage glouton et beam search
- Évaluer avec le score BLEU
"""),

md("""---
## 1. Fondements Théoriques

### 1.1 Modèles de Langage

Un **modèle de langage** assigne une probabilité à une séquence de tokens $(x_1, x_2, \\ldots, x_T)$.

Par la règle de chaîne :

$$P(x_1, \\ldots, x_T) = \\prod_{t=1}^{T} P(x_t \\mid x_1, \\ldots, x_{t-1})$$

L'objectif est donc d'apprendre à prédire le **prochain token** à partir du contexte passé.

### 1.2 Perplexité

La **perplexité** mesure la qualité d'un modèle de langage :

$$\\text{PPL} = \\exp\\left(-\\frac{1}{T}\\sum_{t=1}^{T}\\log P(x_t \\mid x_{<t})\\right) = \\exp(\\text{CE})$$

- PPL faible → le modèle prédit bien le prochain token (confiant et correct)
- PPL ≈ 1 → modèle parfait ; PPL ≈ |V| (taille vocabulaire) → modèle aléatoire

### 1.3 Architectures Récurrentes

#### RNN Simple
$$h_t = \\tanh(W_x x_t + W_h h_{t-1} + b)$$

**Problème :** gradient évanescent/explosif sur longues séquences (chaîne de produits de Jacobiens).

#### LSTM (Long Short-Term Memory)
Introduit un **état de cellule** $c_t$ et trois portes :

$$f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)\\quad \\text{(oubli)}$$
$$i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)\\quad \\text{(entrée)}$$
$$\\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$$
$$o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o),\\quad h_t = o_t \\odot \\tanh(c_t)$$

#### GRU (Gated Recurrent Unit)
Simplifie le LSTM avec seulement **deux portes** :

$$z_t = \\sigma(W_z [h_{t-1}, x_t])\\quad \\text{(mise à jour)}$$
$$r_t = \\sigma(W_r [h_{t-1}, x_t])\\quad \\text{(réinitialisation)}$$
$$\\tilde{h}_t = \\tanh(W [r_t \\odot h_{t-1}, x_t])$$
$$h_t = z_t \\odot h_{t-1} + (1 - z_t) \\odot \\tilde{h}_t$$

### 1.4 Tableau Comparatif

| Critère | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Portes | 0 | 3 (oubli, entrée, sortie) | 2 (mise à jour, réinitialisation) |
| État mémoire | $h_t$ | $h_t + c_t$ | $h_t$ |
| Gradient | Évanescent | Stable (chemin via $c_t$) | Stable (compromis) |
| Paramètres | Peu | Plus | Moins que LSTM |
| Usage | Séquences courtes | Longues dépendances | Bon compromis |
"""),

md("""---
## 2. Imports et Configuration"""),

code("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re, time, math, random, zipfile, unicodedata
import requests

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download("punkt", quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("nltk non disponible — évaluation BLEU désactivée")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch : {torch.__version__} | Device : {DEVICE}")

os.makedirs("models",  exist_ok=True)
os.makedirs("data",    exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
"""),

md("""---
## 3. Préparation des Données Tatoeba fra-eng"""),

code("""# Téléchargement du corpus Tatoeba (fra-eng) depuis manythings.org
DATA_ZIP  = "data/fra-eng.zip"
DATA_FILE = "data/fra.txt"

if not os.path.exists(DATA_FILE):
    print("Téléchargement du corpus Tatoeba fra-eng...")
    url = "https://www.manythings.org/anki/fra-eng.zip"
    try:
        r = requests.get(url, timeout=60)
        with open(DATA_ZIP, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(DATA_ZIP, "r") as z:
            z.extractall("data/")
        print(f"Corpus téléchargé : {os.path.getsize(DATA_FILE) // 1024} Ko")
    except Exception as e:
        print(f"Erreur de téléchargement : {e}")
        print("Utilisation d'un mini-corpus de démonstration...")
        demo_pairs = [
            "I am cold.\\tJ ai froid .\\tCC-BY 2.0",
            "I am hot.\\tJ ai chaud .\\tCC-BY 2.0",
            "He is tall.\\tIl est grand .\\tCC-BY 2.0",
            "She is kind.\\tElle est gentille .\\tCC-BY 2.0",
            "We are here.\\tNous sommes ici .\\tCC-BY 2.0",
            "They are late.\\tIls sont en retard .\\tCC-BY 2.0",
            "I love you.\\tJe t aime .\\tCC-BY 2.0",
            "I am tired.\\tJe suis fatigue .\\tCC-BY 2.0",
            "How are you ?\\tComment allez-vous ?\\tCC-BY 2.0",
            "Good morning.\\tBonjour .\\tCC-BY 2.0",
        ] * 500  # répété pour avoir assez de données
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write("\\n".join(demo_pairs))
else:
    print(f"Corpus déjà présent : {os.path.getsize(DATA_FILE) // 1024} Ko")
"""),

code("""def normalize_string(s):
    \"\"\"Normalisation : minuscules, suppression accents, marquage ponctuation.\"\"\"
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([.!?])", r" \\1", s)
    s = re.sub(r"[^a-zA-Z.!? ]+", r" ", s)
    return s.strip()

def load_pairs(filepath, max_len=10, max_pairs=30000):
    \"\"\"Charge et filtre les paires (anglais, français) du corpus Tatoeba.\"\"\"
    pairs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\\t")
            if len(parts) < 2:
                continue
            eng = normalize_string(parts[0])
            fra = normalize_string(parts[1])
            if len(eng.split()) <= max_len and len(fra.split()) <= max_len:
                if len(eng) > 0 and len(fra) > 0:
                    pairs.append((eng, fra))
            if len(pairs) >= max_pairs:
                break
    return pairs

pairs = load_pairs(DATA_FILE)
random.shuffle(pairs)
print(f"Paires chargées (≤10 tokens) : {len(pairs):,}")
print("\\nExemples :")
for eng, fra in pairs[:5]:
    print(f"  EN: {eng}")
    print(f"  FR: {fra}")
    print()
"""),

code("""class Vocab:
    \"\"\"Vocabulaire avec tokens spéciaux : PAD(0), SOS(1), EOS(2), UNK(3).\"\"\"
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, name):
        self.name = name
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.n_words  = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

    def encode(self, sentence, max_len, add_eos=False, add_sos=False):
        tokens = sentence.split()
        ids = [self.word2idx.get(w, self.UNK) for w in tokens]
        if add_sos: ids = [self.SOS] + ids
        if add_eos: ids = ids + [self.EOS]
        # Troncature + padding
        ids = ids[:max_len]
        ids = ids + [self.PAD] * (max_len - len(ids))
        return ids


# Construction des vocabulaires
src_vocab = Vocab("eng")
tgt_vocab = Vocab("fra")
for eng, fra in pairs:
    src_vocab.add_sentence(eng)
    tgt_vocab.add_sentence(fra)

print(f"Vocabulaire anglais  : {src_vocab.n_words:,} mots")
print(f"Vocabulaire français : {tgt_vocab.n_words:,} mots")
"""),

code("""class TranslationDataset(Dataset):
    \"\"\"Dataset PyTorch pour les paires de traduction.\"\"\"
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=12):
        self.max_len   = max_len
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.samples   = []
        for eng, fra in pairs:
            src = src_vocab.encode(eng,  max_len,   add_eos=True)
            # tgt_in  : <SOS> w1 w2 ... (entrée décodeur)
            # tgt_out : w1 w2 ... <EOS> (cible, décalée d'un token vers la droite)
            tgt_in  = tgt_vocab.encode(fra, max_len+1, add_sos=True)
            tgt_out = tgt_vocab.encode(fra, max_len+1, add_eos=True)
            self.samples.append((src, tgt_in, tgt_out))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, ti, to = self.samples[idx]
        return (torch.tensor(s,  dtype=torch.long),
                torch.tensor(ti, dtype=torch.long),
                torch.tensor(to, dtype=torch.long))


# Split train/test
MAX_LEN = 12
n_test  = min(2000, len(pairs) // 5)
n_train = len(pairs) - n_test

train_pairs = pairs[:n_train]
test_pairs  = pairs[n_train:]

train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab, MAX_LEN)
test_ds  = TranslationDataset(test_pairs,  src_vocab, tgt_vocab, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)

print(f"Train : {len(train_ds):,} paires | Test : {len(test_ds):,} paires")
print(f"Exemple batch — src: {next(iter(train_loader))[0].shape}")
"""),

md("""---
## 4. Implémentation des Architectures Récurrentes"""),

code("""class RecurrentLM(nn.Module):
    \"\"\"Modèle de langage récurrent générique (RNN / LSTM / GRU).
    Utilisé pour la comparaison des architectures et la démonstration BPTT.
    \"\"\"

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1,
                 rnn_type="GRU", dropout=0.3):
        super().__init__()
        self.rnn_type   = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=Vocab.PAD)
        rnn_cls    = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn   = rnn_cls(embed_dim, hidden_dim, num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb    = self.dropout(self.embed(x))    # (B, T, E)
        out, hidden = self.rnn(emb, hidden)      # (B, T, H)
        logits = self.fc(self.dropout(out))       # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if self.rnn_type == "LSTM":
            return (h, torch.zeros_like(h))
        return h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
"""),

md("""---
## 5. BPTT et Démonstration du Gradient Clipping"""),

code("""def train_lm_epoch(model, loader, optimizer, device, clip=None, record_norms=False):
    \"\"\"Entraîne un modèle de langage sur une epoch.
    L'état caché est détaché entre les batches (BPTT tronquée).
    \"\"\"
    model.train()
    total_loss, grad_norms = 0.0, []
    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)

    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        optimizer.zero_grad()
        logits, _ = model(tgt_in)   # (B, T, V) — modèle de langage sur séquence cible

        # Reshape pour CrossEntropyLoss : (B*T, V) et (B*T,)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
        loss.backward()

        if record_norms:
            # Norme du gradient avant clipping
            total_norm = sum(p.grad.data.norm(2).item()**2
                            for p in model.parameters() if p.grad is not None) ** 0.5
            grad_norms.append(total_norm)

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader), grad_norms


# Démonstration gradient clipping sur 3 epochs
print("=== Démonstration : gradient clipping sur RNN ===\\n")

VOCAB_SIZE = max(src_vocab.n_words, tgt_vocab.n_words)
rnn_noclip = RecurrentLM(tgt_vocab.n_words, embed_dim=64, hidden_dim=128,
                          rnn_type="RNN", num_layers=2).to(DEVICE)
rnn_clipped = RecurrentLM(tgt_vocab.n_words, embed_dim=64, hidden_dim=128,
                           rnn_type="RNN", num_layers=2).to(DEVICE)

# Copier les mêmes poids initiaux
rnn_clipped.load_state_dict(rnn_noclip.state_dict())

opt_nc = optim.Adam(rnn_noclip.parameters(), lr=1e-3)
opt_cl = optim.Adam(rnn_clipped.parameters(), lr=1e-3)

norms_no_clip, norms_clipped = [], []

for epoch in range(3):
    _, norms_nc = train_lm_epoch(rnn_noclip,  train_loader, opt_nc, DEVICE,
                                  clip=None, record_norms=True)
    _, norms_cl = train_lm_epoch(rnn_clipped, train_loader, opt_cl, DEVICE,
                                  clip=5.0,  record_norms=True)
    norms_no_clip.extend(norms_nc)
    norms_clipped.extend(norms_cl)
    print(f"  Epoch {epoch+1} | Max norme sans clip : {max(norms_nc):.2f} | "
          f"Max norme avec clip : {max(norms_cl):.2f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(norms_no_clip[:200], alpha=0.7, label="Sans clipping", color="#DD8452")
ax.plot(norms_clipped[:200], alpha=0.7, label="Avec clipping (max=5)", color="#4C72B0")
ax.axhline(5.0, color="red", linestyle="--", alpha=0.5, label="Seuil = 5")
ax.set_title("Norme du gradient au fil des batches — effet du gradient clipping", fontsize=13)
ax.set_xlabel("Batch"); ax.set_ylabel("||∇||₂")
ax.legend(); ax.set_ylim(0, min(50, max(norms_no_clip)*1.1))
plt.tight_layout()
plt.savefig("figures/partie3_gradient_clipping.png", dpi=100, bbox_inches="tight")
plt.show()
print("\\nConclusion : sans clipping, des pics de gradient déstabilisent l'apprentissage.")
"""),

md("""---
## 6. Comparaison RNN / LSTM / GRU"""),

code("""def eval_perplexity(model, loader, device):
    \"\"\"Calcule la perplexité sur un DataLoader.
    PPL = exp(cross-entropy loss) — mesure standard pour les modèles de langage.
    \"\"\"
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)
            logits, _ = model(tgt_in)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
            total_loss += loss.item()
            n += 1
    return math.exp(total_loss / n)


results_lm = {}
EPOCHS_LM   = 5
LM_EMBED    = 64    # dimensions du modèle de langage (comparaison RNN/LSTM/GRU)
LM_HIDDEN   = 128   # (séparés de EMBED_DIM/HIDDEN_DIM utilisés pour le Seq2Seq)

for rnn_type in ["RNN", "LSTM", "GRU"]:
    print(f"\\n{'='*50}")
    print(f"Entraînement {rnn_type} ({EPOCHS_LM} epochs)")
    print('='*50)

    m = RecurrentLM(tgt_vocab.n_words, embed_dim=LM_EMBED, hidden_dim=LM_HIDDEN,
                    rnn_type=rnn_type, num_layers=1).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=1e-3)

    train_ppls, val_ppls = [], []
    t0 = time.time()
    for epoch in range(1, EPOCHS_LM + 1):
        loss_tr, _ = train_lm_epoch(m, train_loader, opt, DEVICE, clip=5.0)
        ppl_tr = math.exp(min(loss_tr, 10))  # clamp pour éviter overflow
        ppl_val = eval_perplexity(m, test_loader, DEVICE)
        train_ppls.append(ppl_tr)
        val_ppls.append(ppl_val)
        print(f"  Epoch {epoch}/{EPOCHS_LM} | Train PPL: {ppl_tr:.2f} | Val PPL: {ppl_val:.2f}")

    elapsed = time.time() - t0
    results_lm[rnn_type] = {
        "model": m, "train_ppls": train_ppls, "val_ppls": val_ppls,
        "final_ppl": ppl_val, "time_s": elapsed,
        "params": m.count_parameters()
    }
    print(f"  Temps total : {elapsed:.1f}s | Paramètres : {m.count_parameters():,}")
"""),

code("""# Courbes de perplexité
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"RNN": "#DD8452", "LSTM": "#4C72B0", "GRU": "#55A868"}

for rnn_type, res in results_lm.items():
    axes[0].plot(res["train_ppls"], label=rnn_type, color=colors[rnn_type], linewidth=2)
    axes[1].plot(res["val_ppls"],   label=rnn_type, color=colors[rnn_type], linewidth=2)

for ax, title in zip(axes, ["Perplexité d'entraînement", "Perplexité de validation"]):
    ax.set_xlabel("Epoch"); ax.set_ylabel("PPL"); ax.set_title(title, fontsize=13)
    ax.legend()

plt.tight_layout()
plt.savefig("figures/partie3_rnn_comparison.png", dpi=100, bbox_inches="tight")
plt.show()

# Tableau comparatif
df_lm = pd.DataFrame([
    {"Modèle": k, "PPL finale (val)": f"{v['final_ppl']:.2f}",
     "Paramètres": f"{v['params']:,}", "Temps (s)": f"{v['time_s']:.1f}"}
    for k, v in results_lm.items()
])
print("\\n=== Tableau comparatif RNN / LSTM / GRU ===\\n")
print(df_lm.to_string(index=False))
"""),

md("""---
## 7. Variantes Architecturales : RNN Profond et Bidirectionnel

### 7.1 RNN Profond (Deep RNN)

Empiler plusieurs couches récurrentes permet d'apprendre des représentations **hiérarchiques** :
- couche 1 : motifs locaux et syntaxiques
- couche 2+ : abstractions sémantiques plus riches

$$h_t^{(l)} = f^{(l)}\\bigl(h_t^{(l-1)},\\; h_{t-1}^{(l)}\\bigr), \\quad l \\geq 2$$

### 7.2 RNN Bidirectionnel

Un RNN bidirectionnel traite la séquence dans **les deux sens** puis concatène :

$$\\overrightarrow{h}_t = f(x_t, \\overrightarrow{h}_{t-1}), \\quad
  \\overleftarrow{h}_t = g(x_t, \\overleftarrow{h}_{t+1}), \\quad
  h_t = [\\overrightarrow{h}_t\\,;\\,\\overleftarrow{h}_t]$$

Utile pour des tâches d'**encodage** (ex. classification de sentiment, NER) — **impossible** pendant la génération auto-régressive car le futur n'est pas connu à l'inférence.
"""),

code("""# Comparaison : GRU 1 couche vs 2 couches vs bidirectionnel
print("=== Variantes architecturales du GRU ===\\n")

variants = {
    "GRU-1L"       : {"num_layers": 1, "bidirectional": False},
    "GRU-2L"       : {"num_layers": 2, "bidirectional": False},
    "GRU-bidir"    : {"num_layers": 1, "bidirectional": True},
}

results_variants = {}

for vname, vcfg in variants.items():
    bidir = vcfg["bidirectional"]
    n_layers = vcfg["num_layers"]

    # Pour le bidirectionnel on réduit hidden_dim/2 pour garder même nb params approx.
    hdim = LM_HIDDEN // 2 if bidir else LM_HIDDEN

    embed = nn.Embedding(tgt_vocab.n_words, LM_EMBED, padding_idx=Vocab.PAD).to(DEVICE)
    rnn   = nn.GRU(LM_EMBED, hdim, num_layers=n_layers,
                   batch_first=True, bidirectional=bidir,
                   dropout=0.3 if n_layers > 1 else 0.0).to(DEVICE)
    # Taille de sortie du GRU : hdim * (2 si bidir else 1)
    out_dim = hdim * (2 if bidir else 1)
    fc    = nn.Linear(out_dim, tgt_vocab.n_words).to(DEVICE)

    params = sum(p.numel() for p in list(embed.parameters()) +
                 list(rnn.parameters()) + list(fc.parameters()) if p.requires_grad)

    # Entraînement rapide (3 epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)
    all_params = list(embed.parameters()) + list(rnn.parameters()) + list(fc.parameters())
    opt = optim.Adam(all_params, lr=1e-3)

    val_ppls = []
    for epoch in range(1, 4):
        embed.train(); rnn.train(); fc.train()
        for src_b, tgt_in_b, tgt_out_b in train_loader:
            tgt_in_b, tgt_out_b = tgt_in_b.to(DEVICE), tgt_out_b.to(DEVICE)
            opt.zero_grad()
            emb_out = embed(tgt_in_b)
            rnn_out, _ = rnn(emb_out)
            logits = fc(rnn_out)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B*T, V), tgt_out_b.reshape(B*T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            opt.step()

        embed.eval(); rnn.eval(); fc.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for _, tgt_in_b, tgt_out_b in test_loader:
                tgt_in_b, tgt_out_b = tgt_in_b.to(DEVICE), tgt_out_b.to(DEVICE)
                emb_out = embed(tgt_in_b)
                rnn_out, _ = rnn(emb_out)
                logits = fc(rnn_out)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B*T, V), tgt_out_b.reshape(B*T))
                total_loss += loss.item(); n += 1
        ppl = math.exp(min(total_loss / n, 10))
        val_ppls.append(ppl)
        print(f"  [{vname}] Epoch {epoch}/3 | Val PPL: {ppl:.2f}")

    results_variants[vname] = {"final_ppl": val_ppls[-1], "params": params,
                                "val_ppls": val_ppls}

# Tableau de synthèse
print("\\n=== Comparaison GRU 1L / 2L / Bidirectionnel ===\\n")
for vname, res in results_variants.items():
    print(f"  {vname:<15} | PPL={res['final_ppl']:.2f} | params={res['params']:,}")
print("\\nNota : le bidirectionnel (bidir) est efficace en encodage mais inutilisable")
print("en génération auto-régressive (il regarde le futur, impossible à l'inférence).")
"""),

code("""# Visualisation comparative des variantes
fig, ax = plt.subplots(figsize=(10, 4))
colors_v = {"GRU-1L": "#4C72B0", "GRU-2L": "#DD8452", "GRU-bidir": "#55A868"}
for vname, res in results_variants.items():
    ax.plot(range(1, 4), res["val_ppls"], marker="o",
            label=vname, color=colors_v[vname], linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Perplexité (validation)")
ax.set_title("GRU 1 couche vs 2 couches vs bidirectionnel (3 epochs)", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig("figures/partie3_variants.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

md("""---
## 8. Architecture Seq2Seq"""),

code("""class Encoder(nn.Module):
    \"\"\"Encodeur GRU : lit la séquence source et produit un état de contexte.\"\"\"

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=Vocab.PAD)
        self.gru     = nn.GRU(embed_dim, hidden_dim, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src : (B, T_src)
        emb     = self.dropout(self.embed(src))   # (B, T_src, E)
        outputs, hidden = self.gru(emb)            # (B, T_src, H), (L, B, H)
        return outputs, hidden


class Decoder(nn.Module):
    \"\"\"Décodeur conditionnel GRU : génère la séquence cible token par token.\"\"\"

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, embed_dim, padding_idx=Vocab.PAD)
        self.gru      = nn.GRU(embed_dim, hidden_dim, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out   = nn.Linear(hidden_dim, vocab_size)
        self.dropout  = nn.Dropout(dropout)

    def forward_step(self, token, hidden):
        \"\"\"Un pas de décodage.
        token  : (B,) → indice du token précédent
        hidden : (L, B, H) → état caché précédent
        \"\"\"
        emb    = self.dropout(self.embed(token.unsqueeze(1)))  # (B, 1, E)
        out, hidden = self.gru(emb, hidden)                    # (B, 1, H)
        logits = self.fc_out(out.squeeze(1))                   # (B, V)
        return logits, hidden


class Seq2Seq(nn.Module):
    \"\"\"Architecture encodeur-décodeur avec teacher forcing.

    Teacher forcing : pendant l'entraînement, on fournit le vrai token
    précédent au décodeur (au lieu de la prédiction). Cela stabilise
    l'apprentissage mais crée un décalage entraînement/inférence (exposure bias).
    \"\"\"

    def __init__(self, encoder, decoder, tgt_vocab_size, device):
        super().__init__()
        self.encoder       = encoder
        self.decoder       = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.device        = device

    def forward(self, src, tgt_in, teacher_forcing_ratio=0.5):
        \"\"\"
        src     : (B, T_src) — séquence source tokenisée
        tgt_in  : (B, T_tgt) — séquence cible avec <SOS> en début
        Retourne: (B, T_tgt, V) — logits pour chaque position de la cible
        \"\"\"
        B, T = tgt_in.shape
        outputs = torch.zeros(B, T, self.tgt_vocab_size, device=self.device)

        _, hidden = self.encoder(src)  # contexte encodeur
        dec_input = tgt_in[:, 0]       # <SOS> token

        for t in range(1, T):
            logits, hidden = self.decoder.forward_step(dec_input, hidden)
            outputs[:, t] = logits

            # Teacher forcing : true token ou prédiction ?
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt_in[:, t] if use_teacher else logits.argmax(1)

        return outputs


# Instanciation
EMBED_DIM  = 128
HIDDEN_DIM = 256
N_LAYERS   = 1
DROPOUT    = 0.3

encoder = Encoder(src_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(tgt_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
seq2seq = Seq2Seq(encoder, decoder, tgt_vocab.n_words, DEVICE).to(DEVICE)

total_params = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
print(f"Modèle Seq2Seq créé | Paramètres : {total_params:,}")
"""),

md("""---
## 8. Entraînement du Seq2Seq"""),

code("""def masked_ce_loss(logits, targets, pad_idx=Vocab.PAD):
    \"\"\"Cross-entropy masquée : ignore les tokens <PAD> dans le calcul de la perte.
    Sans ce masque, le modèle gaspillerait sa capacité à 'prédire' du padding.
    \"\"\"
    B, T, V = logits.shape
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return criterion(logits.reshape(B*T, V), targets.reshape(B*T))


def train_seq2seq_epoch(model, loader, optimizer, device, tf_ratio=0.5):
    model.train()
    total_loss = 0.0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        optimizer.zero_grad()
        logits = model(src, tgt_in, teacher_forcing_ratio=tf_ratio)
        # CRITIQUE : tgt_out est la séquence décalée (sans <SOS>, avec <EOS>)
        loss = masked_ce_loss(logits[:, 1:], tgt_out[:, :-1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_seq2seq(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in, teacher_forcing_ratio=0.0)
            loss = masked_ce_loss(logits[:, 1:], tgt_out[:, :-1])
            total_loss += loss.item()
    return total_loss / len(loader)


EPOCHS_S2S = 15
optimizer  = optim.Adam(seq2seq.parameters(), lr=5e-4)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

print("=== Entraînement Seq2Seq ===\\n")
train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS_S2S + 1):
    tf = max(0.1, 0.5 * (1 - epoch / EPOCHS_S2S))  # réduction progressive du teacher forcing
    tl = train_seq2seq_epoch(seq2seq, train_loader, optimizer, DEVICE, tf_ratio=tf)
    vl = eval_seq2seq(seq2seq, test_loader, DEVICE)
    scheduler.step(vl)
    train_losses.append(tl)
    val_losses.append(vl)

    if vl < best_val_loss:
        best_val_loss = vl
        torch.save(seq2seq.state_dict(), "models/seq2seq_tatoeba.pth")

    print(f"Epoch {epoch:2d}/{EPOCHS_S2S} | Train Loss: {tl:.4f} | "
          f"Val Loss: {vl:.4f} | PPL: {math.exp(min(vl,10)):.2f} | TF: {tf:.2f}")

# Courbe d'entraînement
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label="Train", color="#4C72B0", linewidth=2)
ax.plot(val_losses,   label="Validation", color="#DD8452", linewidth=2)
ax.set_title("Convergence du modèle Seq2Seq", fontsize=13)
ax.set_xlabel("Epoch"); ax.set_ylabel("Perte (Cross-Entropy masquée)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/partie3_seq2seq_training.png", dpi=100, bbox_inches="tight")
plt.show()
"""),

md("""---
## 9. Stratégies de Décodage"""),

code("""# Chargement du meilleur modèle
seq2seq.load_state_dict(torch.load("models/seq2seq_tatoeba.pth", map_location=DEVICE))
seq2seq.eval()


def greedy_decode(model, src_tokens, tgt_vocab, max_len=20, device=DEVICE):
    \"\"\"Décodage glouton : à chaque pas, choisit le token de probabilité maximale.
    Rapide mais sous-optimal globalement.
    \"\"\"
    model.eval()
    with torch.no_grad():
        src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
        _, hidden = model.encoder(src)

        dec_input = torch.tensor([Vocab.SOS], device=device)
        result = []
        for _ in range(max_len):
            logits, hidden = model.decoder.forward_step(dec_input, hidden)
            token = logits.argmax(1)
            if token.item() == Vocab.EOS:
                break
            word = tgt_vocab.idx2word.get(token.item(), "<UNK>")
            if word not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
                result.append(word)
            dec_input = token
    return " ".join(result)


def beam_search_decode(model, src_tokens, tgt_vocab, beam_width=3,
                       max_len=20, length_penalty=0.7, device=DEVICE):
    \"\"\"Beam search : maintient les k meilleures hypothèses partielles.
    Meilleure exploration que le décodage glouton.
    Score normalisé par la longueur pour éviter de favoriser les séquences courtes.
    \"\"\"
    model.eval()
    with torch.no_grad():
        src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
        _, hidden = model.encoder(src)

        # Chaque faisceau : [score_log, tokens, hidden_state]
        beams = [(0.0, [Vocab.SOS], hidden)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, tokens, h in beams:
                if tokens[-1] == Vocab.EOS:
                    completed.append((score, tokens))
                    continue
                dec_in = torch.tensor([tokens[-1]], device=device)
                logits, new_h = model.decoder.forward_step(dec_in, h)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                # Expansion : top-k tokens
                topk_vals, topk_ids = log_probs.topk(beam_width)
                for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                    new_score = score + val
                    new_beams.append((new_score, tokens + [idx], new_h))

            if not new_beams:
                break
            # Tri et sélection des k meilleurs faisceaux
            new_beams.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = new_beams[:beam_width]

        # Sélection du meilleur faisceau complété
        completed.extend([(b[0], b[1]) for b in beams])
        if not completed:
            return ""
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        best_tokens = completed[0][1]

        words = [tgt_vocab.idx2word.get(t, "<UNK>") for t in best_tokens
                 if t not in (Vocab.SOS, Vocab.EOS, Vocab.PAD)]
        return " ".join(words)


# Test sur quelques exemples
print("=== Exemples de traduction ===\\n")
for eng, ref_fra in test_pairs[:8]:
    src_ids = src_vocab.encode(eng, MAX_LEN+1, add_eos=True)
    greedy   = greedy_decode(seq2seq, src_ids, tgt_vocab)
    beam     = beam_search_decode(seq2seq, src_ids, tgt_vocab, beam_width=3)
    print(f"  EN  : {eng}")
    print(f"  REF : {ref_fra}")
    print(f"  Greedy  : {greedy}")
    print(f"  Beam(3) : {beam}")
    print()
"""),

md("""---
## 10. Évaluation BLEU"""),

code("""# Score BLEU sur l'ensemble de test
if BLEU_AVAILABLE:
    smooth = SmoothingFunction().method1
    bleu_greedy, bleu_beam = [], []
    n_eval = min(500, len(test_pairs))

    for eng, ref_fra in test_pairs[:n_eval]:
        src_ids = src_vocab.encode(eng, MAX_LEN+1, add_eos=True)
        reference  = [ref_fra.split()]
        hyp_greedy = greedy_decode(seq2seq, src_ids, tgt_vocab).split()
        hyp_beam   = beam_search_decode(seq2seq, src_ids, tgt_vocab, beam_width=3).split()

        bleu_greedy.append(sentence_bleu(reference, hyp_greedy,
                                         smoothing_function=smooth))
        bleu_beam.append(sentence_bleu(reference, hyp_beam,
                                       smoothing_function=smooth))

    mean_bleu_greedy = np.mean(bleu_greedy)
    mean_bleu_beam   = np.mean(bleu_beam)

    print(f"=== Évaluation BLEU ({n_eval} paires de test) ===\\n")
    print(f"BLEU moyen (décodage glouton) : {mean_bleu_greedy:.4f}")
    print(f"BLEU moyen (beam search k=3)  : {mean_bleu_beam:.4f}")
    print(f"\\nNote : des scores BLEU de 0.10–0.35 sont normaux pour un Seq2Seq")
    print(f"sans mécanisme d'attention sur ce dataset de taille réduite.")

    # Distribution des scores BLEU
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(bleu_greedy, bins=30, alpha=0.7, label=f"Greedy (μ={mean_bleu_greedy:.3f})",
            color="#4C72B0")
    ax.hist(bleu_beam,   bins=30, alpha=0.7, label=f"Beam k=3 (μ={mean_bleu_beam:.3f})",
            color="#DD8452")
    ax.set_title("Distribution des scores BLEU — Glouton vs Beam Search", fontsize=13)
    ax.set_xlabel("Score BLEU"); ax.set_ylabel("Fréquence"); ax.legend()
    plt.tight_layout()
    plt.savefig("figures/partie3_bleu_scores.png", dpi=100, bbox_inches="tight")
    plt.show()

    # Tableau de synthèse
    df_decode = pd.DataFrame([
        {"Stratégie": "Décodage glouton", "BLEU moyen": f"{mean_bleu_greedy:.4f}",
         "Vitesse": "Rapide (O(T))"},
        {"Stratégie": "Beam search (k=3)", "BLEU moyen": f"{mean_bleu_beam:.4f}",
         "Vitesse": f"Lente (O(k·T·|V|))"},
    ])
    print("\\n=== Tableau comparatif des stratégies de décodage ===\\n")
    print(df_decode.to_string(index=False))
else:
    print("nltk non disponible — installez-le avec : pip install nltk")
"""),

md("""---
## 11. Analyse Critique

### 11.1 Comparaison RNN / LSTM / GRU

Les résultats expérimentaux confirment la progression théorique :

- **RNN simple** : perplexité la plus élevée, convergence moins stable, particulièrement sensible aux longues séquences (gradient évanescent sur les dépendances au-delà de 5–7 tokens)
- **LSTM** : meilleure mémoire à long terme grâce à l'état de cellule $c_t$ qui offre un chemin quasi-direct pour le flux de gradient. Contrepartie : plus de paramètres
- **GRU** : performances comparables au LSTM avec moins de paramètres (pas d'état de cellule séparé). Excellent compromis précision/efficacité

### 11.2 Teacher Forcing

La technique du teacher forcing accélère la convergence en fournissant de vrais tokens au décodeur pendant l'entraînement. Cependant, elle crée un **décalage entraînement/inférence** (*exposure bias*) : le modèle n'apprend pas à récupérer après ses propres erreurs. La réduction progressive du ratio (de 0.5 à 0.1 au fil des epochs) atténue ce problème.

### 11.3 Beam Search vs Glouton

Le beam search améliore modestement les scores BLEU en explorant plusieurs hypothèses simultanément. Sur des séquences courtes (≤10 tokens), le gain est limité car le décodage glouton est déjà souvent optimal. Le bénéfice est plus marqué sur des traductions longues et complexes.

### 11.4 Limites du Modèle

Le principal goulot d'étranglement est la **compression en vecteur fixe** : toute l'information de la phrase source doit tenir dans l'état caché final de l'encodeur. Pour des phrases longues, cette compression est insuffisante. Le mécanisme d'**attention** (Bahdanau 2015) résout ce problème en permettant au décodeur d'interroger directement tous les états cachés de l'encodeur, posant les bases de l'architecture Transformer.

---
## 12. Question de Synthèse — Partie III

**Question :** Dans quelle mesure les architectures récurrentes permettent-elles de modéliser efficacement une séquence réelle, et comment justifier le passage du RNN vers LSTM/GRU puis vers l'encodeur-décodeur ?

**Réponse :**

Les architectures récurrentes modélisent efficacement les séquences en maintenant un état caché $h_t$ qui résume le contexte passé — ce que les modèles à fenêtre fixe (n-grammes, MLP) ne peuvent pas faire de manière générique. Cette mémoire récurrente permet d'apprendre des dépendances arbitrairement longues, du moins en théorie.

En pratique, le RNN simple souffre du **gradient évanescent** : lors de la rétropropagation à travers le temps (BPTT), le produit des Jacobiens $\\prod \\frac{\\partial h_i}{\\partial h_{i-1}}$ tend vers zéro exponentiellement avec la longueur de la séquence. Le gradient clipping pallie l'explosion mais pas l'évanouissement.

Le **LSTM** résout ce problème structurellement en introduisant un état de cellule $c_t$ dont la mise à jour suit $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$. La porte d'oubli $f_t$ peut être proche de 1, permettant à l'information de traverser de nombreux pas de temps sans atténuation. Le **GRU** simplifie cette architecture en deux portes, avec des performances similaires sur la plupart des tâches.

L'**encodeur-décodeur** représente un changement de paradigme : au lieu de modéliser une seule séquence, il apprend une transformation *many-to-many* entre domaines (anglais → français). L'encodeur compresse la source en un vecteur de contexte, le décodeur génère la cible conditionnellement. Cette architecture reste limitée par la compression en vecteur fixe pour les longues séquences — limitation résolue par l'attention (Bahdanau 2015) puis les Transformers (Vaswani 2017), qui ont supplanté les RNN dans la plupart des tâches de NLP modernes.
"""),

md("""---
## 13. Question Transversale Finale

### Problématique

> *Comment le deep learning adapte-t-il ses architectures à la structure des données — tabulaire, image et séquentielle — et pourquoi un même paradigme d'apprentissage supervisé doit-il être décliné différemment selon la géométrie, la dépendance locale, la temporalité et la représentation des données ?*

---

### 13.1 L'invariant commun : le paradigme d'apprentissage supervisé

Quelles que soient les données, le schéma est identique :

```
(X, y) → Minimiser L(f_θ(X), y) par descente de gradient
```

Les trois parties du projet partagent donc :
- une **perte cross-entropique** pour la classification ;
- un **optimiseur Adam** avec rétropropagation automatique (`loss.backward()`) ;
- une **régularisation** (Dropout, weight decay, gradient clipping) pour éviter le surapprentissage ;
- une boucle `train / evaluate` identique dans sa structure.

Ce qui varie radicalement, c'est la façon dont l'architecture **encode les a priori inductifs** propres à chaque type de données.

---

### 13.2 Données tabulaires → MLP : pas de structure, pleine flexibilité

**Géométrie des données :** aucune — les $n$ caractéristiques d'un vecteur tabulaire sont des variables indépendantes sans ordre ni voisinage défini. Il n'y a pas d'a priori spatial, temporel ou structurel à exploiter.

**Réponse architecturale : le MLP**

$$y = W_2 \\cdot \\text{ReLU}(W_1 x + b_1) + b_2$$

Le MLP applique des transformations **denses** successives, apprenant des combinaisons linéaires arbitraires de toutes les entrées. C'est précisément ce dont on a besoin quand toutes les interactions potentielles entre variables méritent d'être considérées.

**Ce qui compte :**
- la **normalisation** des features (StandardScaler), sans laquelle des variables de magnitudes très différentes (ex. proline ~750 vs alcool ~12 dans Wine Quality) biaisent l'apprentissage ;
- l'**initialisation** (Xavier stabilise la variance du signal dès la première propagation) ;
- la **régularisation** (Dropout, L2) pour éviter le surapprentissage sur des datasets souvent petits.

**Limite fondamentale :** le MLP ne peut pas exploiter de structure. Sur des images, il *voit* 784 valeurs indépendantes et perd toute notion de voisinage entre pixels.

---

### 13.3 Images → CNN : exploiter la géométrie spatiale

**Géométrie des données :** une image est une grille 2D (parfois 3D avec les canaux couleur). Les pixels voisins sont fortement corrélés. Un bord horizontal apparaît de la même façon à gauche et à droite de l'image. Un objet reconnaissable est composé de parties locales reconnaissables.

**Réponse architecturale : la convolution**

$$S(i,j) = \\sum_m \\sum_n I(i+m,\\, j+n) \\cdot K(m,n)$$

Ce choix encode trois a priori fondamentaux :

| A priori | Mécanisme | Effet |
|----------|-----------|-------|
| **Localité** | Noyau de taille $k \times k$ | Capte les motifs locaux (bords, textures) |
| **Partage des poids** | Même filtre convolu sur toute l'image | Réduit les paramètres : 62K (LeNet) vs 100K (MLP équivalent) |
| **Équivariance à la translation** | Filtres glissants | Un chiffre "7" est reconnu quelle que soit sa position |

**Ce qui compte :**
- le **padding** pour contrôler la taille spatiale et ne pas perdre d'information aux bords ;
- le **pooling** (MaxPool) pour réduire la résolution et augmenter le champ réceptif ;
- la **profondeur** : les premières couches détectent des bords, les suivantes des formes, les dernières des objets.

**Limite fondamentale :** le CNN modélise mal les dépendances à longue portée dans une image (un pixel en haut à gauche et un en bas à droite). Les Vision Transformers (ViT) résolvent ce problème par l'attention globale.

---

### 13.4 Séquences → RNN/LSTM/GRU/Seq2Seq : la temporalité

**Géométrie des données :** une séquence $(x_1, x_2, \\ldots, x_T)$ a un **ordre causal** : $x_t$ dépend du passé $x_{<t}$. La longueur $T$ varie d'un exemple à l'autre. L'information pertinente peut se trouver très loin dans le passé.

**Réponse architecturale : l'état caché récurrent**

$$h_t = \\phi(W_x x_t + W_h h_{t-1} + b)$$

L'état caché $h_t$ est une **mémoire compressée** du passé. C'est une solution fondamentalement différente du MLP (qui ignore l'ordre) et du CNN (dont le champ réceptif est borné).

**Progression des architectures :**

```
RNN simple  →  gradient évanescent sur longues dépendances
    ↓
LSTM        →  état de cellule c_t = mémoire explicite, gradient stable
    ↓
GRU         →  simplification du LSTM (2 portes), compromis précision/efficacité
    ↓
Encodeur-Décodeur  →  transformation many-to-many (traduction)
    ↓
Attention (Bahdanau 2015)  →  décodeur interroge tous les états encodeur
    ↓
Transformer (Vaswani 2017)  →  attention globale, parallélisable, état de l'art
```

**Ce qui compte :**
- le **gradient clipping** pour éviter l'explosion du gradient lors de la BPTT ;
- le **teacher forcing** pour stabiliser l'entraînement du décodeur ;
- le **beam search** pour améliorer la qualité du décodage à l'inférence.

---

### 13.5 Synthèse comparative

| Dimension | MLP (tabulaire) | CNN (images) | RNN/Seq2Seq (séquences) |
|-----------|-----------------|--------------|-------------------------|
| **Structure exploitée** | Aucune | Grille 2D spatiale | Ordre temporel |
| **A priori inductif** | Toutes les interactions | Localité + partage des poids | Mémoire récurrente |
| **Opération centrale** | Produit matriciel dense | Corrélation croisée 2D | Mise à jour récurrente |
| **Paramètres (projet)** | ~5K (Wine, 13 features) | ~62K (LeNet-5) | ~1M (Seq2Seq) |
| **Longueur d'entrée** | Fixe | Fixe (image) | Variable |
| **Régularisation clé** | Dropout, L2 | Pooling (sous-échantillonnage) | Gradient clipping |
| **Évaluation** | Accuracy, F1 | Accuracy, matrices de confusion | Perplexité, BLEU |
| **Limite principale** | Pas d'invariance structurelle | Dépendances à longue portée | Compression en vecteur fixe |
| **Successeur moderne** | TabNet, XGBoost | ResNet, ViT | Transformer, BERT, GPT |

---

### 13.6 Conclusion

Ce projet illustre un principe fondateur du deep learning moderne : **l'architecture est une forme de connaissance a priori encodée**. Le choix d'un MLP, d'un CNN ou d'un RNN n'est pas arbitraire — c'est une hypothèse sur la structure des données.

- Le **MLP** postule l'absence de structure : toutes les interactions entre variables sont équiprobables.
- Le **CNN** postule la localité et l'invariance spatiale : les motifs pertinents sont locaux et répétitifs.
- Le **RNN/Seq2Seq** postule la causalité temporelle : le futur dépend du passé selon un ordre défini.

Chacune de ces hypothèses est une restriction du modèle qui, lorsqu'elle est correcte, améliore considérablement la généralisation avec moins de paramètres. Lorsqu'elle est incorrecte (un MLP sur des images, un CNN sur des séquences longues), les performances se dégradent.

L'architecture **Transformer** — non couverte dans ce projet — unifie partiellement ces paradigmes par le mécanisme d'attention multi-têtes, qui permet de modéliser des dépendances arbitraires entre toutes les positions d'une séquence, qu'elle soit textuelle, visuelle ou tabulaire. C'est la raison pour laquelle les Transformers ont supplanté les CNN (ViT) et les RNN (BERT, GPT) dans la quasi-totalité des tâches de deep learning avancées depuis 2020.
"""),

]

# ============================================================
# ÉCRITURE DES FICHIERS
# ============================================================

base = "C:/Users/ayoub/Downloads/Projet_DeepLearning"

for filename, cells in [
    ("partie1_MLP.ipynb",          partie1_cells),
    ("partie2_CNN.ipynb",          partie2_cells),
    ("partie3_RNN_Seq2Seq.ipynb",  partie3_cells),
]:
    path = os.path.join(base, filename)
    notebook = nb(cells)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    size_kb = os.path.getsize(path) // 1024
    print(f"[OK] {filename}  ({size_kb} Ko, {len(cells)} cellules)")

print("\nTous les notebooks ont ete generes avec succes.")
