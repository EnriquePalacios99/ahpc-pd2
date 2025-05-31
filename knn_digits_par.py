#!/usr/bin/env python
"""
knn_digits_par.py  –  Versión paralela de K‑NN (clasificación de dígitos) con:
  • Directivas MPI (bcast, scatter, gather)
  • Medición de tiempos: distribución, cómputo, recolección, total
  • Accuracy del modelo
  • Dataset original *digits* o sintético vía make_classification
  • Guarda dos tipos de figuras en disco (headless):
        1. PCA 2‑D etiquetas reales vs predicciones
        2. (opcional) matriz de confusión como heatmap

Ejecución rápida
----------------
Dataset *digits* (1 437 train, 360 test) – 3 ranks:
    mpiexec -n 3 python knn_digits_par.py

Dataset sintético (50 000 train, 5 000 test) – 8 ranks – k = 5:
    mpiexec -n 8 python knn_digits_par.py 50000 5

CLI
---
argv[1]  n_train  (opcional)  → genera datos sintéticos si se indica
argv[2]  k        (opcional)  → vecinos (default = 3)

Todos los resultados se imprimen en consola y las imágenes se guardan en:
    pca_knn.png, cm_knn.png
"""

# ─────────────────── Imports ───────────────────
from mpi4py import MPI
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import Counter
import sys
import os

# backend headless para que funcione en terminal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ───────────── Configuración MPI ──────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ───────────── Leer argumentos CLI ────────────
train_size_cli = int(sys.argv[1]) if len(sys.argv) > 1 else None
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# ──────── Dataset: digits o sintético ─────────
if rank == 0:
    if train_size_cli is None:
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
    else:
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=train_size_cli,
            n_features=64,
            n_informative=40,
            n_redundant=0,
            n_classes=10,
            random_state=42,
        )

    # test_size conforme al enunciado (10 %)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    n_features = X_train.shape[1]
else:
    X_train = X_test = y_train = y_test = None
    n_features = 64

# ───────── Broadcast del set de prueba ────────
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

n_test = X_test.shape[0]

# total entrenamiento + padding si es necesario
if rank == 0:
    n_train_total = X_train.shape[0]
else:
    n_train_total = None
n_train_total = comm.bcast(n_train_total, root=0)

remainder = n_train_total % size if rank == 0 else None
remainder = comm.bcast(remainder, root=0)

if remainder != 0 and rank == 0:
    pad = size - remainder
    X_pad = np.zeros((pad, n_features), dtype=X_train.dtype)
    y_pad = np.zeros(pad, dtype=np.int32)
    X_train = np.vstack([X_train, X_pad])
    y_train = np.hstack([y_train.astype(np.int32), y_pad])
    n_train_total += pad

n_train_total = comm.bcast(n_train_total, root=0)

a_per_proc = n_train_total // size

# buffers locales
recvbuf_X = np.empty(a_per_proc * n_features, dtype=np.float64)
local_y = np.empty(a_per_proc, dtype=np.int32)

# ────────────────── Scatter ───────────────────
T0 = MPI.Wtime()
if rank == 0:
    comm.Scatter(np.ascontiguousarray(X_train.ravel()), recvbuf_X, root=0)
    comm.Scatter(np.ascontiguousarray(y_train.astype(np.int32)), local_y, root=0)
else:
    comm.Scatter(None, recvbuf_X, root=0)
    comm.Scatter(None, local_y, root=0)
T_dist = MPI.Wtime()

local_X = recvbuf_X.reshape(a_per_proc, n_features)

# ───────────── Cálculo local de vecinos ───────

def euclidean(batch, point):
    return np.sqrt(np.sum((batch - point) ** 2, axis=1))

local_preds = []
for p_tst in X_test:
    d = euclidean(local_X, p_tst)
    idx = d.argsort()[:k]
    local_preds.append((d[idx], local_y[idx]))
T_comp = MPI.Wtime()

# ─────────────── Gather en root ───────────────
all_preds = comm.gather(local_preds, root=0)
T_gather = MPI.Wtime()

# ─────────────── Post‑proceso root ────────────
if rank == 0:
    final_preds = []
    for i in range(n_test):
        neigh = []
        for proc in all_preds:
            neigh.extend(zip(proc[i][0], proc[i][1]))
        neigh.sort(key=lambda x: x[0])
        top_k = [lab for _, lab in neigh[:k]]
        final_preds.append(Counter(top_k).most_common(1)[0][0])

    final_preds = np.array(final_preds, dtype=np.int32)
    acc = np.mean(final_preds == y_test)

    # ────── Consola ──────
    print(f"[Ranks: {size}] n_train={n_train_total} n_test={n_test} k={k}")
    print(f"Total      : {T_gather - T0:8.4f} s")
    print(f"  Distrib  : {T_dist   - T0:8.4f} s")
    print(f"  Cómputo  : {T_comp   - T_dist:8.4f} s")
    print(f"  Recolec  : {T_gather - T_comp:8.4f} s")
    print(f"Accuracy   : {acc:8.4f}\n")

    # ────── Métricas extra ──────
    cm = confusion_matrix(y_test, final_preds)
    print("Confusion matrix\n", cm)
    print("\nClassification report\n", classification_report(y_test, final_preds, digits=4))

    # ──────────── Imágenes ────────────
    out_dir = os.getcwd()

    # 1) PCA scatter
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X_test)
    plt.figure(figsize=(10, 4))
    for idx, (title, labels) in enumerate([
        ("Etiquetas reales", y_test),
        ("Predicciones", final_preds),
    ]):
        plt.subplot(1, 2, idx + 1)
        plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels, s=15, cmap="tab10")
        plt.title(title)
        plt.axis("equal")
    plt.tight_layout()
    pca_path = os.path.join(out_dir, "pca_knn.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()

    # 2) Heatmap de la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de confusión")
    cm_path = os.path.join(out_dir, "cm_knn.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"\nImágenes guardadas: {pca_path}, {cm_path}")
