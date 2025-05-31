"""
Script para comparar rendimiento entre implementación secuencial y paralela
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

def euclidean_distance(a, b):
    """Versión vectorizada para cálculo secuencial"""
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict_sequential(test_point, X_train, y_train, k):
    """Predicción KNN secuencial"""
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

def run_sequential_experiment(multiplier=1, k=3):
    """Ejecutar experimento secuencial"""
    # Cargar datos
    digits = load_digits()

    # Multiplicar dataset
    if multiplier > 1:
        X_expanded = np.tile(digits.data, (multiplier, 1))
        y_expanded = np.tile(digits.target, multiplier)
    else:
        X_expanded = digits.data
        y_expanded = digits.target

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_expanded, y_expanded, test_size=0.2, random_state=42
    )

    print(f"Experimento secuencial - Multiplicador: {multiplier}")
    print(f"Entrenamiento: {len(X_train)}, Prueba: {len(X_test)}")

    # Medir tiempo
    start_time = time.time()

    # Predicciones
    y_pred = []
    for i, x in enumerate(X_test):
        pred = knn_predict_sequential(x, X_train, y_train, k)
        y_pred.append(pred)

        # Progreso cada 10%
        if (i + 1) % (len(X_test) // 10) == 0:
            progress = (i + 1) / len(X_test) * 100
            print(f"  Progreso: {progress:.0f}%")

    end_time = time.time()
    total_time = end_time - start_time

    # Calcular precisión
    accuracy = np.mean(np.array(y_pred) == y_test)

    print(f"  Tiempo total: {total_time:.4f} seg")
    print(f"  Precisión: {accuracy:.4f}")

    return {
        'multiplier': multiplier,
        'total_time': total_time,
        'accuracy': accuracy,
        'dataset_size': len(X_expanded)
    }

def load_parallel_results():
    """Cargar resultados paralelos para comparación"""
    try:
        df = pd.read_csv('knn_results_analysis.csv')
        return df
    except FileNotFoundError:
        print("⚠️  No se encontraron resultados paralelos")
        print("Ejecuta primero analyze_results.py")
        return None

def compare_implementations():
    """Comparar implementaciones secuencial y paralela"""
    print("=== COMPARACIÓN SECUENCIAL vs PARALELO ===")

    # Multiplicadores a probar
    multipliers = [1, 2, 4]

    # Ejecutar experimentos secuenciales
    sequential_results = []
    for mult in multipliers:
        print(f"\nEjecutando experimento secuencial mult={mult}...")
        result = run_sequential_experiment(mult)
        sequential_results.append(result)

    seq_df = pd.DataFrame(sequential_results)

    # Cargar resultados paralelos
    par_df = load_parallel_results()

    if par_df is None:
        print("No se pueden hacer comparaciones sin resultados paralelos")
        return

    # Análisis comparativo
    plt.figure(figsize=(15, 10))

    # Subplot 1: Tiempos vs Multiplicador
    plt.subplot(2, 3, 1)
    plt.plot(seq_df['multiplier'], seq_df['total_time'], 'ro-',
             linewidth=3, markersize=8, label='Secuencial')

    # Promediar tiempos paralelos por multiplicador
    par_avg = par_df.groupby('multiplier')['total_time'].mean().reset_index()
    plt.plot(par_avg['multiplier'], par_avg['total_time'], 'bo-',
             linewidth=3, markersize=8, label='Paralelo (promedio)')

    plt.xlabel('Multiplicador de Dataset')
    plt.ylabel('Tiempo Total (seg)')
    plt.title('Tiempo de Ejecución: Secuencial vs Paralelo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot 2: Speedup vs Procesos para cada multiplicador
    plt.subplot(2, 3, 2)
    for mult in par_avg['multiplier']:
        if mult in seq_df['multiplier'].values:
            t_seq = seq_df[seq_df['multiplier'] == mult]['total_time'].iloc[0]
            subset = par_df[par_df['multiplier'] == mult]
            speedup = t_seq / subset['total_time']
            plt.plot(subset['processes'], speedup, 'o-',
                     label=f'Mult {mult}', linewidth=2, markersize=6)

    # Línea ideal
    max_p = par_df['processes'].max()
    plt.plot([1, max_p], [1, max_p], 'k--', alpha=0.5, label='Ideal')

    plt.xlabel('Número de Procesos')
    plt.ylabel('Speedup')
    plt.title('Speedup Relativo al Secuencial')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Eficiencia
    plt.subplot(2, 3, 3)
    for mult in par_avg['multiplier']:
        if mult in seq_df['multiplier'].values:
            t_seq = seq_df[seq_df['multiplier'] == mult]['total_time'].iloc[0]
            subset = par_df[par_df['multiplier'] == mult]
            speedup = t_seq / subset['total_time']
            efficiency = speedup / subset['processes']
            plt.plot(subset['processes'], efficiency, 'o-',
                     label=f'Mult {mult}', linewidth=2, markersize=6)

    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Ideal')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Eficiencia')
    plt.title('Eficiencia Relativa al Secuencial')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 4: Precisión
    plt.subplot(2, 3, 4)
    plt.plot(seq_df['multiplier'], seq_df['accuracy'], 'ro-',
             linewidth=3, markersize=8, label='Secuencial')

    par_acc_avg = par_df.groupby('multiplier')['accuracy'].mean().reset_index()
    plt.plot(par_acc_avg['multiplier'], par_acc_avg['accuracy'], 'bo-',
             linewidth=3, markersize=8, label='Paralelo (promedio)')

    plt.xlabel('Multiplicador de Dataset')
    plt.ylabel('Precisión (Accuracy)')
    plt.title('Comparación de Precisión')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.95, 1.0])

    # Subplot 5: Tiempo por muestra
    plt.subplot(2, 3, 5)
    seq_time_per_sample = seq_df['total_time'] / (seq_df['dataset_size'] * 0.2)  # 20% para test
    plt.plot(seq_df['multiplier'], seq_time_per_sample, 'ro-',
             linewidth=3, markersize=8, label='Secuencial')

    # Mejor tiempo paralelo por multiplicador
    par_best = par_df.groupby('multiplier')['total_time'].min().reset_index()
    par_best['samples'] = par_best['multiplier'] * 1797 * 0.2
    par_best['time_per_sample'] = par_best['total_time'] / par_best['samples']
    plt.plot(par_best['multiplier'], par_best['time_per_sample'], 'go-',
             linewidth=3, markersize=8, label='Paralelo (mejor)')

    plt.xlabel('Multiplicador de Dataset')
    plt.ylabel('Tiempo por Muestra de Prueba (seg)')
    plt.title('Eficiencia de Procesamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot 6: Resumen de mejoras
    plt.subplot(2, 3, 6)
    improvements = []
    for mult in par_avg['multiplier']:
        if mult in seq_df['multiplier'].values:
            t_seq = seq_df[seq_df['multiplier'] == mult]['total_time'].iloc[0]
            t_par_best = par_df[par_df['multiplier'] == mult]['total_time'].min()
            improvement = t_seq / t_par_best
            improvements.append(improvement)
        else:
            improvements.append(1)

    plt.bar(par_avg['multiplier'], improvements, alpha=0.7, color='green')
    plt.xlabel('Multiplicador de Dataset')
    plt.ylabel('Factor de Mejora')
    plt.title('Mejora Máxima con Paralelización')
    plt.grid(True, alpha=0.3)

    # Agregar valores en las barras
    for i, v in enumerate(improvements):
        plt.text(par_avg['multiplier'].iloc[i], v + 0.1, f'{v:.1f}x',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('sequential_vs_parallel_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Resumen numérico
    print("\n=== RESUMEN DE COMPARACIÓN ===")
    print("Multiplicador | Tiempo Seq | Mejor Par | Mejora | Efic. Máx")
    print("-" * 60)
    for mult in par_avg['multiplier']:
        if mult in seq_df['multiplier'].values:
            t_seq = seq_df[seq_df['multiplier'] == mult]['total_time'].iloc[0]
            subset = par_df[par_df['multiplier'] == mult]
            t_par_best = subset['total_time'].min()
            best_p = subset.loc[subset['total_time'].idxmin(), 'processes']
            improvement = t_seq / t_par_best
            max_efficiency = improvement / best_p

            print(f"{mult:^11} | {t_seq:^10.3f} | {t_par_best:^9.3f} | {improvement:^6.1f}x | {max_efficiency:^9.3f}")

    # Guardar datos para referencia
    seq_df.to_csv('sequential_results.csv', index=False)
    print(f"\n✅ Resultados secuenciales guardados en 'sequential_results.csv'")
    print("✅ Gráfico guardado como 'sequential_vs_parallel_comparison.png'")

if __name__ == "__main__":
    compare_implementations()