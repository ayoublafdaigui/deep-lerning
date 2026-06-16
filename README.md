# Projet Deep Learning — EMSI Casablanca 2025–2026

Projet final du module Deep Learning couvrant trois architectures fondamentales : MLP, CNN et RNN/Seq2Seq, implémentées en PyTorch sur des jeux de données réels.

---

## Idée générale

Ce projet explore de manière progressive les grandes familles de réseaux de neurones profonds :

- **Partie 1 — MLP** : Perceptron multicouche pour la classification de vins (Wine dataset, scikit-learn). Comparaison de plusieurs configurations d'initialisation (Xavier, Gaussienne, Constante), régularisation L2, early stopping sur validation set, évaluation finale sur test set.
- **Partie 2 — CNN** : Réseau convolutif (LeNet-5) pour la classification d'images (MNIST). Calcul analytique des dimensions de sortie, expériences sur padding/stride/pooling/filtres/convolution 1×1, comparaison de 6 variantes architecturales.
- **Partie 3 — RNN / Seq2Seq** : Modèle de langage (RNN/LSTM/GRU) avec mesure de perplexité, puis architecture Seq2Seq encodeur-décodeur avec teacher forcing pour la traduction français→anglais (corpus Tatoeba). Décodage glouton, beam search, score BLEU.

---

## Structure du dépôt

```
Projet_DeepLearning/
│
├── partie1_MLP.ipynb            # Notebook Partie 1 — MLP (Wine dataset)
├── partie2_CNN.ipynb            # Notebook Partie 2 — CNN/LeNet-5 (MNIST)
├── partie3_RNN_Seq2Seq.ipynb    # Notebook Partie 3 — RNN, LSTM, GRU, Seq2Seq
│
├── generate_notebooks.py        # Script Python générateur des trois notebooks
│
├── requirements.txt             # Dépendances Python
│
├── data/                        # Données téléchargées automatiquement au runtime
│   └── (fra-eng corpus Tatoeba, téléchargé à l'exécution)
│
├── models/                      # Poids sauvegardés après entraînement (.pt)
│
├── figures/                     # Graphiques générés (courbes loss/accuracy, etc.)
│
├── Projet_Deep-Learning_EMSI.pdf       # Sujet du projet
├── fiche_synthese_MLP_PyTorch.pdf      # Fiche synthèse MLP
└── synthese_rnn_seq2seq.pdf            # Fiche synthèse RNN/Seq2Seq
```

---

## Prérequis et installation

```bash
pip install -r requirements.txt
```

Versions principales : Python ≥ 3.9, PyTorch ≥ 2.0, torchvision ≥ 0.15, scikit-learn ≥ 1.3, NLTK ≥ 3.8.

---

## Utilisation

### Option A — Générer les notebooks depuis le script

```bash
python generate_notebooks.py
```

Cela (re)génère les trois fichiers `.ipynb` dans le répertoire courant.

### Option B — Exécuter directement les notebooks

Ouvrir chaque notebook dans Jupyter Lab / VS Code et exécuter les cellules dans l'ordre. Les données MNIST et le corpus Tatoeba sont téléchargés automatiquement à la première exécution.

---

## Points clés par partie

| Partie | Dataset | Modèle | Métriques |
|--------|---------|--------|-----------|
| 1 — MLP | Wine (13 features, 3 classes) | MLP personnalisable | Accuracy (val + test), Loss |
| 2 — CNN | MNIST (28×28, 10 classes) | LeNet-5 + 5 variantes | Accuracy, nb paramètres |
| 3 — RNN/Seq2Seq | PTB / Tatoeba fra-eng | RNN·LSTM·GRU + Seq2Seq | Perplexité, BLEU score |

---

## Auteur

**Ayoub Lafdaigui** — EMSI Casablanca, 2025–2026
