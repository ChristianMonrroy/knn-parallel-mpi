from mpi4py import MPI
import numpy as np
from collections import Counter
import sys
import time

# Solo el rank 0 importará estas librerías para visualización
if MPI.COMM_WORLD.Get_rank() == 0:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    """Calcula la distancia euclidiana vectorizada entre un punto y un conjunto de puntos"""
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# Configuración MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parámetros
k = 3

# Obtener el multiplicador del dataset desde línea de comandos
if len(sys.argv) >= 2:
    data_multiplier = int(sys.argv[1])
else:
    data_multiplier = 1

# Cargar y preparar datos (solo en rank 0)
if rank == 0:
    # Cargar dataset de dígitos
    digits = load_digits()

    # Multiplicar los datos para escalabilidad
    if data_multiplier > 1:
        X_expanded = np.tile(digits.data, (data_multiplier, 1))
        y_expanded = np.tile(digits.target, data_multiplier)
    else:
        X_expanded = digits.data
        y_expanded = digits.target

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_expanded, y_expanded, test_size=0.2, random_state=42
    )

    train_size = len(X_train)
    test_size = len(X_test)
    num_features = X_train.shape[1]

    print(f"Dataset multiplicado por: {data_multiplier}")
    print(f"Tamaño de entrenamiento: {train_size}")
    print(f"Tamaño de prueba: {test_size}")
    print(f"Número de características: {num_features}")
    print(f"Número de procesos: {size}")

else:
    X_train = y_train = X_test = y_test = None
    train_size = test_size = num_features = None

# Broadcast de información necesaria
train_size = comm.bcast(train_size, root=0)
test_size = comm.bcast(test_size, root=0)
num_features = comm.bcast(num_features, root=0)

# Broadcast de datos de prueba
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# Calcular tamaño local para cada proceso
local_train_size = train_size // size
remainder = train_size % size

# Ajustar para procesos que reciben datos extra
if rank < remainder:
    local_train_size += 1

# Preparar arrays locales
local_X = np.empty((local_train_size, num_features), dtype='float64')
local_y = np.empty(local_train_size, dtype='int')

# Timing - inicio
t_start = MPI.Wtime()

# Distribuir datos de entrenamiento usando Scatterv para manejar distribución desigual
if rank == 0:
    # Crear listas de tamaños y desplazamientos para Scatterv
    sendcounts_X = []
    sendcounts_y = []
    displs_X = []
    displs_y = []

    offset = 0
    for i in range(size):
        local_size = train_size // size
        if i < remainder:
            local_size += 1

        sendcounts_X.append(local_size * num_features)
        sendcounts_y.append(local_size)
        displs_X.append(offset * num_features)
        displs_y.append(offset)
        offset += local_size
else:
    sendcounts_X = sendcounts_y = displs_X = displs_y = None

# Scatterv para distribuir datos
comm.Scatterv([X_train, sendcounts_X, displs_X, MPI.DOUBLE], local_X, root=0)
comm.Scatterv([y_train, sendcounts_y, displs_y, MPI.INT], local_y, root=0)

t_dist = MPI.Wtime()

# Cómputo local - calcular distancias y encontrar k vecinos más cercanos locales
local_predictions = []
for i, x in enumerate(X_test):
    # Calcular distancias a todos los puntos de entrenamiento locales
    dists = euclidean_distance(local_X, x)

    # Encontrar los k vecinos más cercanos locales
    k_indices = dists.argsort()[:min(k, len(dists))]
    k_distances = dists[k_indices]
    k_labels = local_y[k_indices]

    # Guardar distancias y etiquetas de los k vecinos más cercanos locales
    local_predictions.append(list(zip(k_distances, k_labels)))

t_comp = MPI.Wtime()

# Gather todas las predicciones locales en el proceso raíz
all_predictions = comm.gather(local_predictions, root=0)

t_gather = MPI.Wtime()

# Predicción final en rank 0
if rank == 0:
    final_predictions = []

    for i in range(test_size):
        # Combinar todos los vecinos de todos los procesos para esta muestra de prueba
        all_neighbors = []
        for proc_predictions in all_predictions:
            all_neighbors.extend(proc_predictions[i])

        # Ordenar por distancia y tomar los k más cercanos globalmente
        all_neighbors.sort(key=lambda x: x[0])
        top_k_labels = [label for _, label in all_neighbors[:k]]

        # Votación mayoritaria
        most_common = Counter(top_k_labels).most_common(1)
        final_predictions.append(most_common[0][0])

    final_predictions = np.array(final_predictions)
    accuracy = np.mean(final_predictions == y_test)

    # Cálculo de tiempos
    total_time = t_gather - t_start
    distribution_time = t_dist - t_start
    computation_time = t_comp - t_dist
    communication_time = t_gather - t_comp

    # Resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Procesos: {size}")
    print(f"Multiplicador de datos: {data_multiplier}")
    print(f"Tiempo total: {total_time:.4f} sec")
    print(f"  - Distribución: {distribution_time:.4f} sec")
    print(f"  - Cómputo: {computation_time:.4f} sec")
    print(f"  - Comunicación: {communication_time:.4f} sec")
    print(f"Precisión (Accuracy): {accuracy:.4f}")

    # Guardar resultados en archivo
    with open(f'results_p{size}_mult{data_multiplier}.txt', 'w') as f:
        f.write(f"{size}\t{total_time:.4f}\t{distribution_time:.4f}\t{computation_time:.4f}\t{communication_time:.4f}\t{accuracy:.4f}\n")

    # Visualización de algunas predicciones
    if data_multiplier == 1:  # Solo para dataset original
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(X_test):
                # Reshape para mostrar imagen 8x8
                ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
                ax.set_title(f"Pred: {final_predictions[i]}\nTrue: {y_test[i]}")
                ax.axis('off')

        plt.suptitle(f"Predicciones KNN Paralelo (p={size})")
        plt.tight_layout()
        plt.savefig(f'predictions_p{size}.png', dpi=150, bbox_inches='tight')
        plt.show()

    print(f"Resultados guardados en: results_p{size}_mult{data_multiplier}.txt")
    if data_multiplier == 1:
        print(f"Gráfico guardado en: predictions_p{size}.png")