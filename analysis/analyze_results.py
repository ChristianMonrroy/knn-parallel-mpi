import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from collections import defaultdict

def load_results():
    """Cargar todos los archivos de resultados"""
    results = []

    # Buscar archivos de resultados
    result_files = glob.glob('results/results_p*_mult*.txt')

    for file in result_files:
        # Extraer parámetros del nombre del archivo
        filename = os.path.basename(file)
        parts = filename.replace('results_p', '').replace('.txt', '').split('_mult')

        if len(parts) == 2:
            processes = int(parts[0])
            multiplier = int(parts[1])

            # Leer datos
            with open(file, 'r') as f:
                line = f.readline().strip()
                if line:
                    data = line.split('\t')
                    if len(data) >= 6:
                        total_time = float(data[1])
                        dist_time = float(data[2])
                        comp_time = float(data[3])
                        comm_time = float(data[4])
                        accuracy = float(data[5])

                        results.append({
                            'processes': processes,
                            'multiplier': multiplier,
                            'dataset_size': multiplier * 1797,  # Tamaño base del dataset digits
                            'total_time': total_time,
                            'distribution_time': dist_time,
                            'computation_time': comp_time,
                            'communication_time': comm_time,
                            'accuracy': accuracy
                        })

    return pd.DataFrame(results)

def calculate_speedup_efficiency(df):
    """Calcular speedup y eficiencia"""
    df_calc = df.copy()

    # Agrupar por multiplicador para calcular speedup
    for mult in df_calc['multiplier'].unique():
        mask = df_calc['multiplier'] == mult
        subset = df_calc[mask].copy()

        # Tiempo secuencial (p=1)
        t_seq = subset[subset['processes'] == 1]['total_time']
        if len(t_seq) > 0:
            t_seq = t_seq.iloc[0]

            # Calcular speedup y eficiencia
            df_calc.loc[mask, 'speedup'] = t_seq / df_calc.loc[mask, 'total_time']
            df_calc.loc[mask, 'efficiency'] = df_calc.loc[mask, 'speedup'] / df_calc.loc[mask, 'processes']

    return df_calc

def plot_execution_times(df):
    """Gráfico de tiempos de ejecución"""
    plt.figure(figsize=(15, 10))

    # Subplot 1: Tiempo total vs procesos
    plt.subplot(2, 3, 1)
    for mult in sorted(df['multiplier'].unique()):
        subset = df[df['multiplier'] == mult].sort_values('processes')
        plt.plot(subset['processes'], subset['total_time'], 'o-',
                 label=f'Mult. {mult} (N={mult*1797})', linewidth=2, markersize=6)
    plt.xlabel('Número de Procesos')
    plt.ylabel('Tiempo Total (seg)')
    plt.title('Tiempo de Ejecución vs Procesos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot 2: Desglose de tiempos
    plt.subplot(2, 3, 2)
    mult_ref = 4  # Multiplicador de referencia
    subset = df[df['multiplier'] == mult_ref].sort_values('processes')

    plt.plot(subset['processes'], subset['computation_time'], 'o-',
             label='Cómputo', linewidth=2, markersize=6)
    plt.plot(subset['processes'], subset['communication_time'], 's-',
             label='Comunicación', linewidth=2, markersize=6)
    plt.plot(subset['processes'], subset['distribution_time'], '^-',
             label='Distribución', linewidth=2, markersize=6)

    plt.xlabel('Número de Procesos')
    plt.ylabel('Tiempo (seg)')
    plt.title(f'Desglose de Tiempos (Mult. {mult_ref})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Subplot 3: Speedup
    plt.subplot(2, 3, 4)
    for mult in sorted(df['multiplier'].unique()):
        if 'speedup' in df.columns:
            subset = df[df['multiplier'] == mult].sort_values('processes')
            if not subset['speedup'].isna().all():
                plt.plot(subset['processes'], subset['speedup'], 'o-',
                         label=f'Mult. {mult}', linewidth=2, markersize=6)

    # Línea ideal
    max_p = df['processes'].max()
    plt.plot([1, max_p], [1, max_p], 'k--', alpha=0.5, label='Ideal')

    plt.xlabel('Número de Procesos')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Procesos')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 4: Eficiencia
    plt.subplot(2, 3, 5)
    for mult in sorted(df['multiplier'].unique()):
        if 'efficiency' in df.columns:
            subset = df[df['multiplier'] == mult].sort_values('processes')
            if not subset['efficiency'].isna().all():
                plt.plot(subset['processes'], subset['efficiency'], 'o-',
                         label=f'Mult. {mult}', linewidth=2, markersize=6)

    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Ideal')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Eficiencia')
    plt.title('Eficiencia vs Procesos')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 5: Precisión
    plt.subplot(2, 3, 3)
    for mult in sorted(df['multiplier'].unique()):
        subset = df[df['multiplier'] == mult].sort_values('processes')
        plt.plot(subset['processes'], subset['accuracy'], 'o-',
                 label=f'Mult. {mult}', linewidth=2, markersize=6)

    plt.xlabel('Número de Procesos')
    plt.ylabel('Precisión (Accuracy)')
    plt.title('Precisión vs Procesos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.9, 1.0])

    # Subplot 6: Escalabilidad débil
    plt.subplot(2, 3, 6)
    # Filtrar datos donde multiplier ≈ processes (escalabilidad débil)
    weak_scaling = []
    for _, row in df.iterrows():
        if abs(row['multiplier'] - row['processes']) <= 1:  # Tolerancia
            weak_scaling.append(row)

    if weak_scaling:
        weak_df = pd.DataFrame(weak_scaling).sort_values('processes')
        plt.plot(weak_df['processes'], weak_df['total_time'], 'ro-',
                 linewidth=2, markersize=8, label='Escalabilidad Débil')
        plt.xlabel('Número de Procesos')
        plt.ylabel('Tiempo Total (seg)')
        plt.title('Análisis de Escalabilidad Débil')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig('knn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def theoretical_analysis(df):
    """Análisis teórico vs experimental"""
    print("=== ANÁLISIS TEÓRICO VS EXPERIMENTAL ===")

    # Parámetros teóricos (basados en la presentación)
    # T(n,p) = a + (n_tr * n_te * k) / p + b * log(p)

    mult_ref = 4
    subset = df[df['multiplier'] == mult_ref].sort_values('processes')

    if len(subset) > 0:
        n_tr = mult_ref * 1797 * 0.8  # 80% para entrenamiento
        n_te = mult_ref * 1797 * 0.2  # 20% para prueba
        k = 3

        print(f"Parámetros del análisis:")
        print(f"  - Datos de entrenamiento: {int(n_tr)}")
        print(f"  - Datos de prueba: {int(n_te)}")
        print(f"  - k: {k}")

        # Ajustar modelo teórico a datos experimentales
        # T(p) = C1 + C2/p + C3*log(p)

        processes = subset['processes'].values
        times = subset['total_time'].values

        # Crear matriz para ajuste por mínimos cuadrados
        A = np.column_stack([
            np.ones(len(processes)),  # Constante
            1.0 / processes,          # Término computacional
            np.log(processes)         # Término de comunicación
        ])

        # Resolver sistema
        coeffs = np.linalg.lstsq(A, times, rcond=None)[0]
        C1, C2, C3 = coeffs

        print(f"Coeficientes ajustados:")
        print(f"  - C1 (constante): {C1:.4f}")
        print(f"  - C2 (cómputo): {C2:.4f}")
        print(f"  - C3 (comunicación): {C3:.4f}")

        # Predicciones teóricas
        theoretical_times = C1 + C2/processes + C3*np.log(processes)

        # Calcular número óptimo de procesos
        # dT/dp = -C2/p² + C3/p = 0
        # p_opt = sqrt(C2/C3)
        if C3 > 0:
            p_opt = np.sqrt(C2/C3)
            print(f"Número óptimo de procesos (teórico): {p_opt:.1f}")

        # Gráfico comparativo
        plt.figure(figsize=(10, 6))
        plt.plot(processes, times, 'ro-', label='Experimental', markersize=8, linewidth=2)
        plt.plot(processes, theoretical_times, 'b--', label='Teórico ajustado', linewidth=2)

        if C3 > 0 and p_opt <= max(processes):
            t_opt = C1 + C2/p_opt + C3*np.log(p_opt)
            plt.axvline(x=p_opt, color='g', linestyle=':', alpha=0.7, label=f'Óptimo teórico (p≈{p_opt:.1f})')
            plt.plot(p_opt, t_opt, 'go', markersize=10)

        plt.xlabel('Número de Procesos')
        plt.ylabel('Tiempo Total (seg)')
        plt.title('Comparación Teórica vs Experimental')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig('theoretical_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Error relativo
        rel_error = np.abs(theoretical_times - times) / times * 100
        print(f"Error relativo promedio: {np.mean(rel_error):.2f}%")

def scalability_analysis(df):
    """Análisis de escalabilidad fuerte y débil"""
    print("\n=== ANÁLISIS DE ESCALABILIDAD ===")

    # Escalabilidad fuerte (dataset fijo)
    print("Escalabilidad Fuerte:")
    mult_fixed = 4
    strong_scaling = df[df['multiplier'] == mult_fixed].sort_values('processes')

    if len(strong_scaling) > 1:
        initial_time = strong_scaling.iloc[0]['total_time']
        final_time = strong_scaling.iloc[-1]['total_time']
        initial_p = strong_scaling.iloc[0]['processes']
        final_p = strong_scaling.iloc[-1]['processes']

        expected_speedup = final_p / initial_p
        actual_speedup = initial_time / final_time
        strong_efficiency = actual_speedup / expected_speedup

        print(f"  - Dataset fijo: multiplicador {mult_fixed}")
        print(f"  - Procesos: {initial_p} → {final_p}")
        print(f"  - Speedup esperado: {expected_speedup:.2f}")
        print(f"  - Speedup real: {actual_speedup:.2f}")
        print(f"  - Eficiencia de escalabilidad fuerte: {strong_efficiency:.3f}")

    # Escalabilidad débil (datos proporcionales a procesos)
    print("\nEscalabilidad Débil:")
    weak_data = []
    for _, row in df.iterrows():
        if abs(row['multiplier'] - row['processes']) <= 1:
            weak_data.append(row)

    if len(weak_data) > 1:
        weak_df = pd.DataFrame(weak_data).sort_values('processes')
        time_variation = (weak_df['total_time'].max() - weak_df['total_time'].min()) / weak_df['total_time'].min()

        print(f"  - Variación temporal: {time_variation:.3f}")
        print(f"  - Escalabilidad débil {'BUENA' if time_variation < 0.2 else 'REGULAR' if time_variation < 0.5 else 'MALA'}")

        # Condición teórica: n_tr ~ p*log(p)
        for _, row in weak_df.iterrows():
            p = row['processes']
            n_tr_actual = row['dataset_size'] * 0.8
            n_tr_theoretical = p * np.log(p) * 1000  # Factor de escala
            print(f"  - p={p}: n_tr={int(n_tr_actual)}, teórico∝{p*np.log(p):.1f}")

def accuracy_analysis(df):
    """Análisis de precisión"""
    print("\n=== ANÁLISIS DE PRECISIÓN ===")

    # Verificar variación de precisión con número de procesos
    for mult in sorted(df['multiplier'].unique()):
        subset = df[df['multiplier'] == mult].sort_values('processes')
        if len(subset) > 1:
            acc_mean = subset['accuracy'].mean()
            acc_std = subset['accuracy'].std()
            acc_min = subset['accuracy'].min()
            acc_max = subset['accuracy'].max()

            print(f"Multiplicador {mult}:")
            print(f"  - Precisión promedio: {acc_mean:.4f}")
            print(f"  - Desviación estándar: {acc_std:.4f}")
            print(f"  - Rango: [{acc_min:.4f}, {acc_max:.4f}]")

            if acc_std > 0.01:
                print(f"  - ⚠️  Variación significativa detectada")
            else:
                print(f"  - ✅ Precisión estable")

    # Análisis de causas de variación
    print("\nPosibles causas de variación en precisión:")
    print("1. Distribución desigual de datos entre procesos")
    print("2. Diferencias en el orden de procesamiento")
    print("3. Efectos de punto flotante en cálculos paralelos")
    print("4. Variaciones en la selección de k-vecinos por empates en distancias")

def generate_report(df):
    """Generar reporte completo"""
    print("\n" + "="*60)
    print("REPORTE COMPLETO - KNN PARALELO CON MPI")
    print("="*60)

    print(f"Experimentos realizados: {len(df)}")
    print(f"Multiplicadores probados: {sorted(df['multiplier'].unique())}")
    print(f"Números de procesos: {sorted(df['processes'].unique())}")

    # Mejor configuración por criterio
    best_speedup = df.loc[df['speedup'].idxmax()] if 'speedup' in df.columns and not df['speedup'].isna().all() else None
    best_efficiency = df.loc[df['efficiency'].idxmax()] if 'efficiency' in df.columns and not df['efficiency'].isna().all() else None
    fastest_time = df.loc[df['total_time'].idxmin()]

    print("\nMejores configuraciones:")
    if best_speedup is not None:
        print(f"- Mayor speedup: p={int(best_speedup['processes'])}, mult={int(best_speedup['multiplier'])}, S={best_speedup['speedup']:.2f}")
    if best_efficiency is not None:
        print(f"- Mayor eficiencia: p={int(best_efficiency['processes'])}, mult={int(best_efficiency['multiplier'])}, E={best_efficiency['efficiency']:.3f}")
    print(f"- Menor tiempo: p={int(fastest_time['processes'])}, mult={int(fastest_time['multiplier'])}, T={fastest_time['total_time']:.4f}s")

    # Recomendaciones
    print("\nRecomendaciones:")
    if len(df[df['processes'] <= 4]) > 0:
        small_p = df[df['processes'] <= 4]
        if not small_p.empty and 'efficiency' in small_p.columns:
            avg_eff_small = small_p['efficiency'].mean()
            if avg_eff_small > 0.7:
                print("✅ Buen rendimiento para pocos procesos (p≤4)")
            else:
                print("⚠️  Rendimiento limitado para pocos procesos")

    if len(df[df['processes'] >= 8]) > 0:
        large_p = df[df['processes'] >= 8]
        if not large_p.empty and 'efficiency' in large_p.columns:
            avg_eff_large = large_p['efficiency'].mean()
            if avg_eff_large > 0.5:
                print("✅ Escalabilidad aceptable para muchos procesos (p≥8)")
            else:
                print("⚠️  Escalabilidad limitada para muchos procesos")

def main():
    """Función principal de análisis"""
    print("Cargando resultados de experimentos KNN...")

    # Verificar si existe el directorio de resultados
    if not os.path.exists('results'):
        print("❌ No se encontró el directorio 'results'")
        print("Asegúrate de ejecutar primero los experimentos con run_experiments.sh")
        return

    # Cargar datos
    df = load_results()

    if df.empty:
        print("❌ No se encontraron archivos de resultados")
        print("Archivos esperados: results/results_p*_mult*.txt")
        return

    print(f"✅ Cargados {len(df)} experimentos")
    print(df[['processes', 'multiplier', 'total_time', 'accuracy']].head())

    # Calcular métricas derivadas
    df = calculate_speedup_efficiency(df)

    # Realizar análisis
    plot_execution_times(df)
    theoretical_analysis(df)
    scalability_analysis(df)
    accuracy_analysis(df)
    generate_report(df)

    # Guardar DataFrame para análisis posterior
    df.to_csv('knn_results_analysis.csv', index=False)
    print(f"\n✅ Resultados guardados en 'knn_results_analysis.csv'")
    print("✅ Gráficos guardados como 'knn_analysis.png' y 'theoretical_comparison.png'")

if __name__ == "__main__":
    main()