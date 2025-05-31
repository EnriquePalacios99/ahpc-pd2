from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Cargar y dividir los datos
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Parámetro
k = 3

# Medir tiempo de ejecución
start_time = time.time()

# Realizar predicciones
y_pred = [knn_predict(x, X_train, y_train, k) for x in X_test]

# Evaluar
accuracy = np.mean(y_pred == y_test)
end_time = time.time()

print(f"Accuracy: {accuracy:.4f}")
print(f"Execution time (sequential): {end_time - start_time:.4f} sec")

import matplotlib.pyplot as plt
# Visualize 10 predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    ax.axis('off')
plt.suptitle("Sample Predictions (Sequential KNN)")
plt.tight_layout()
plt.show()
