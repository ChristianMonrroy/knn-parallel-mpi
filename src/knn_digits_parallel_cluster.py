from mpi4py import MPI
import numpy as np
from collections import Counter
import sys
import time

# Solo el rank 0 importará estas librerías para visualización
if MPI.COMM_WORLD.Get_rank() == 0:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

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

# ============================================================================
# �� PARÁMETRO CRÍTICO: k_max (máximo vecinos por proceso)
# ============================================================================
# Estrategia optimizada: balance entre corrección y eficiencia
k_max = max(k * size * 2, 50)  # Suficientes candidatos, pero controlado
k_max = min(k_max, 200)        # Límite superior para evitar overflow

if rank == 0:
    print(f"�� CONFIGURACIÓN OPTIMIZADA:")
    print(f"   k (vecinos finales): {k}")
    print(f"   k_max (por proceso): {k_max}")
    print(f"   Total candidatos: {k_max * size}")
    print(f"   Procesos: {size}")

# Cargar y preparar datos (solo en rank 0)
if rank == 0:
    # Cargar dataset de dígitos
    digits = load_digits()

    # Multiplicar los datos para escalabilidad
    if data_multiplier > 1:
        X_expanded = np.tile(digits.data, (data_multiplier, 1))
        y_expanded = np.tile(digits.target, data_multiplier)
        
        # Agregar ruido mínimo para simular variabilidad real
        noise_std = 0.01
        for i in range(1, data_multiplier):
            start_idx = i * len(digits.data)
            end_idx = (i + 1) * len(digits.data)
            noise = np.random.normal(0, noise_std, digits.data.shape)
            X_expanded[start_idx:end_idx] += noise
    else:
        X_expanded = digits.data
        y_expanded = digits.target

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_expanded, y_expanded, test_size=0.2, random_state=42, stratify=y_expanded
    )

    train_size = len(X_train)
    test_size = len(X_test)
    num_features = X_train.shape[1]

    print(f"Dataset multiplicado por: {data_multiplier}")
    print(f"Tamaño de entrenamiento: {train_size}")
    print(f"Tamaño de prueba: {test_size}")
    print(f"Número de características: {num_features}")
    print(f"Número de procesos: {size}")
    
    # Verificar que k_max no exceda el tamaño local
    local_train_size_check = train_size // size
    if k_max > local_train_size_check:
        k_max = local_train_size_check
        print(f"⚠️ k_max ajustado a {k_max} (tamaño local máximo)")

else:
    X_train = y_train = X_test = y_test = None
    train_size = test_size = num_features = None

# Broadcast de información necesaria
train_size = comm.bcast(train_size, root=0)
test_size = comm.bcast(test_size, root=0)
num_features = comm.bcast(num_features, root=0)
k_max = comm.bcast(k_max, root=0)

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

# ============================================================================
# �� CÓMPUTO OPTIMIZADO: k_max vecinos por proceso
# ============================================================================
local_predictions = []

# Procesar por lotes para manejar datasets muy grandes
batch_size = min(100, test_size)  # Procesar máximo 100 puntos por vez
num_batches = (test_size + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, test_size)
    batch_X_test = X_test[start_idx:end_idx]
    
    batch_predictions = []
    
    for i, x in enumerate(batch_X_test):
        # Calcular distancias a todos los puntos de entrenamiento locales
        dists = euclidean_distance(local_X, x)
        
        # ✅ ESTRATEGIA OPTIMIZADA: 
        # Tomar los k_max vecinos más cercanos locales (no todos)
        effective_k_max = min(k_max, len(dists))
        k_max_indices = dists.argsort()[:effective_k_max]
        k_max_distances = dists[k_max_indices]
        k_max_labels = local_y[k_max_indices]
        
        # Guardar k_max mejores vecinos locales con info de proceso para debug
        local_candidates = []
        for dist, label in zip(k_max_distances, k_max_labels):
            local_candidates.append((dist, label, rank))
        
        batch_predictions.append(local_candidates)
    
    local_predictions.extend(batch_predictions)

t_comp = MPI.Wtime()

# Gather todas las predicciones locales en el proceso raíz
if rank == 0:
    print(f"�� Estadísticas de comunicación:")
    print(f"   Candidatos por proceso: {k_max}")
    print(f"   Total candidatos por punto: {k_max * size}")
    print(f"   Puntos de test: {test_size}")
    print(f"   Datos a transferir: ~{k_max * size * test_size:,} valores")

all_predictions = comm.gather(local_predictions, root=0)

t_gather = MPI.Wtime()

# Predicción final en rank 0
if rank == 0:
    final_predictions = []
    debug_info = []

    for i in range(test_size):
        # Combinar todos los vecinos de todos los procesos para esta muestra de prueba
        all_neighbors = []
        for proc_predictions in all_predictions:
            all_neighbors.extend(proc_predictions[i])

        # Ordenar por distancia y tomar los k más cercanos globalmente
        all_neighbors.sort(key=lambda x: x[0])
        top_k_global = all_neighbors[:k]
        top_k_labels = [label for _, label, _ in top_k_global]

        # Votación mayoritaria
        most_common = Counter(top_k_labels).most_common(1)
        final_predictions.append(most_common[0][0])
        
        # Guardar info de debug para el primer punto
        if i == 0:
            debug_info = top_k_global

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
    print(f"k_max usado: {k_max}")
    print(f"Tiempo total: {total_time:.4f} sec")
    print(f"  - Distribución: {distribution_time:.4f} sec")
    print(f"  - Cómputo: {computation_time:.4f} sec")
    print(f"  - Comunicación: {communication_time:.4f} sec")
    print(f"Precisión (Accuracy): {accuracy:.4f}")

    # Debug del primer punto de test
    print(f"\n�� DEBUG - Primer punto de test:")
    print(f"Etiqueta real: {y_test[0]}")
    print(f"Predicción: {final_predictions[0]}")
    print(f"Top-{k} vecinos globales:")
    for j, (dist, label, proc_rank) in enumerate(debug_info):
        print(f"  {j+1}. Distancia: {dist:.4f}, Etiqueta: {label}, Proceso: {proc_rank}")

    # Verificación de la solución
    if size == 1:
        print(f"\n=== BASELINE REFERENCE ===")
        print(f"Accuracy baseline (1 proceso): {accuracy:.4f}")
        print(f"Esta precisión debería mantenerse similar en configuraciones paralelas")
    else:
        print(f"\n=== VERIFICACIÓN ===")
        print(f"Compara esta accuracy con la baseline de 1 proceso")
        
    # Análisis de errores
    errors = final_predictions != y_test
    error_rate = np.mean(errors) * 100
    print(f"Tasa de error: {error_rate:.2f}%")

    # Guardar resultados en archivo
    with open(f'results_p{size}_mult{data_multiplier}.txt', 'w') as f:
        f.write(f"{size}\t{total_time:.4f}\t{distribution_time:.4f}\t{computation_time:.4f}\t{communication_time:.4f}\t{accuracy:.4f}\n")

    print(f'Resultados guardados en: results_p{size}_mult{data_multiplier}.txt')

# ============================================================================
# �� REPORTE FINAL DE OPTIMIZACIÓN
# ============================================================================
if rank == 0:
    print(f"\n�� REPORTE DE OPTIMIZACIÓN:")
    print(f"================================")
    print(f"✅ Algoritmo corregido: Selección correcta de k-vecinos globales")
    print(f"⚡ Comunicación optimizada: {k_max} vecinos por proceso (vs todas las distancias)")
    print(f"�� Escalabilidad mejorada: Funciona con datasets grandes")
    print(f"�� Balance: Precisión correcta + Eficiencia práctica")