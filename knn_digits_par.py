#!/usr/bin/env python
"""
knn_digits_par.py  –  Versión paralela de K-NN usando mpi4py.

Ejecución típica (dataset digits):
    mpiexec --oversubscribe -n 3 python knn_digits_par.py

Ejecución con datos sintéticos (50 000 muestras):
    mpiexec -n 8 python knn_digits_par.py 50000

CLI
---
    argv[1]  (opcional)  n_train  → genera make_classification si se indica
    argv[2]  (opcional)  k        → vecinos (default = 3)

Salida en consola
-----------------
    • nº de procesos, tamaños de train/test, k
    • tiempos: total, distribución, cómputo, recolección
    • accuracy, matriz de confusión, clasificación
    • primeros 10 ejemplos y_pred vs y_true
"""

from mpi4py import MPI
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import sys

# ──────────────────────────────────── MPI setup ─────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ─────────────────────────────── Argumentos CLI ────────────────────────────────
train_size_cli = int(sys.argv[1]) if len(sys.argv) > 1 else None
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# ───────────────────────────── Carga / generación datos ────────────────────────
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_features = X_train.shape[1]
else:
    X_train = X_test = y_train = y_test = None
    n_features = 64  # reserva

# ─────────────────────── Broadcast de test y metadatos ─────────────────────────
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)
n_test = X_test.shape[0]

# tamaño total entrenamiento y padding si no es múltiplo de p
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

# volver a emitir n_train_total tras el padding
n_train_total = comm.bcast(n_train_total, root=0)
n_train_local = n_train_total // size

# ─────────────────────────── Distribución (Scatter) ────────────────────────────
T0 = MPI.Wtime()

# X_train: enviamos/recibimos como vector 1-D contiguo
if rank == 0:
    sendbuf_X = np.ascontiguousarray(X_train.ravel())
else:
    sendbuf_X = None
recvbuf_X = np.empty(n_train_local * n_features, dtype=np.float64)
comm.Scatter(sendbuf_X, recvbuf_X, root=0)
local_X = recvbuf_X.reshape(n_train_local, n_features)

# y_train: int32 contiguo
if rank == 0:
    sendbuf_y = np.ascontiguousarray(y_train.astype(np.int32))
else:
    sendbuf_y = None
local_y = np.empty(n_train_local, dtype=np.int32)
comm.Scatter(sendbuf_y, local_y, root=0)

T_dist = MPI.Wtime()

# ────────────────────────────── Cómputo local ──────────────────────────────────
def euclidean_distance(batch, point):
    return np.sqrt(np.sum((batch - point) ** 2, axis=1))

local_preds = []
for point in X_test:
    dists = euclidean_distance(local_X, point)
    k_idx = dists.argsort()[:k]
    local_preds.append((dists[k_idx], local_y[k_idx]))

T_comp = MPI.Wtime()

# ───────────────────────────── Recolección global ──────────────────────────────
all_preds = comm.gather(local_preds, root=0)
T_gather = MPI.Wtime()

# ─────────────────────────── Resultados (solo rank 0) ──────────────────────────
if rank == 0:
    final_preds = []
    for i in range(n_test):
        neighbours = []
        for proc in all_preds:
            neighbours.extend(zip(proc[i][0], proc[i][1]))
        neighbours.sort(key=lambda x: x[0])
        top_k = [lab for _, lab in neighbours[:k]]
        final_preds.append(Counter(top_k).most_common(1)[0][0])

    final_preds = np.array(final_preds, dtype=np.int32)
    accuracy = np.mean(final_preds == y_test)

    print(f"[Ranks: {size}] n_train={n_train_total} n_test={n_test} k={k}")
    print(f"Total      : {T_gather - T0:8.4f} s")
    print(f"  Distrib  : {T_dist   - T0:8.4f} s")
    print(f"  Cómputo  : {T_comp   - T_dist:8.4f} s")
    print(f"  Recolec  : {T_gather - T_comp:8.4f} s")
    print(f"Accuracy   : {accuracy:8.4f}\n")

    # métricas adicionales en texto
    print("Confusion matrix")
    print(confusion_matrix(y_test, final_preds))
    print("\nClassification report")
    print(classification_report(y_test, final_preds, digits=4))

    print("\nEjemplos (y_true → y_pred):")
    for yt, yp in zip(y_test[:10], final_preds[:10]):
        print(f"    {yt} → {yp}")

    # Guardar figura opcional sin servidor X
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_2D = pca.fit_transform(X_test)

        plt.figure(figsize=(10, 4))
        for idx, (title, labels) in enumerate(
            [("Etiquetas reales", y_test), ("Predicciones", final_preds)]
        ):
            plt.subplot(1, 2, idx + 1)
            plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels, s=15, cmap="tab10")
            plt.title(title)
            plt.axis("equal")
        plt.tight_layout()
        plt.savefig("knn_digits_par.png", dpi=150)
        print("\n(Figura guardada como knn_digits_par.png)")
    except Exception as exc:
        print(f"\n(Figura omitida: {exc})")
