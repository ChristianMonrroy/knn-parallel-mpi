#!/usr/bin/env python3

import glob
import numpy as np

def analyze_validation():
    print("🔍 ANÁLISIS DE VALIDACIÓN")
    print("=" * 40)
    
    # Buscar archivos de resultados
    result_files = glob.glob("results_p*_mult1.txt")
    
    if not result_files:
        print("❌ No se encontraron archivos de resultados")
        print("   Espera a que terminen los jobs y ejecuta de nuevo")
        return
    
    accuracies = []
    processes = []
    times = []
    
    print("\n📊 RESULTADOS:")
    print("Procesos | Accuracy | Tiempo (s) | Estado")
    print("-" * 45)
    
    for file in sorted(result_files):
        try:
            with open(file, 'r') as f:
                line = f.readline().strip()
                parts = line.split('\t')
                
                p = int(parts[0])
                time_total = float(parts[1])
                accuracy = float(parts[5])
                
                processes.append(p)
                accuracies.append(accuracy)
                times.append(time_total)
                
                # Determinar estado
                if len(accuracies) == 1:
                    status = "BASELINE"
                else:
                    diff = abs(accuracy - accuracies[0])
                    if diff < 0.01:  # Menos de 1% de diferencia
                        status = "✅ OK"
                    elif diff < 0.05:  # Menos de 5%
                        status = "⚠️ DUDA"
                    else:
                        status = "❌ ERROR"
                
                print(f"   {p:2d}    | {accuracy:6.4f}  |  {time_total:6.3f}   | {status}")
                
        except Exception as e:
            print(f"⚠️ Error leyendo {file}: {e}")
    
    if len(accuracies) >= 2:
        print("\n🎯 ANÁLISIS DE CORRECCIÓN:")
        baseline_acc = accuracies[0]
        max_diff = max([abs(acc - baseline_acc) for acc in accuracies])
        
        print(f"   Accuracy baseline (p=1): {baseline_acc:.4f}")
        print(f"   Máxima diferencia: {max_diff:.6f}")
        
        if max_diff < 0.001:  # Menos de 0.1%
            print("   🎉 CORRECCIÓN EXITOSA: Accuracy prácticamente idéntica")
        elif max_diff < 0.01:  # Menos de 1%
            print("   ✅ CORRECCIÓN BUENA: Diferencias mínimas aceptables")
        elif max_diff < 0.05:  # Menos de 5%
            print("   ⚠️ POSIBLE PROBLEMA: Diferencias moderadas")
        else:
            print("   ❌ CORRECCIÓN FALLIDA: Diferencias significativas")
            
        # Análisis de speedup
        if len(times) >= 2:
            print(f"\n⚡ SPEEDUP:")
            baseline_time = times[0]
            for i, (p, t) in enumerate(zip(processes, times)):
                speedup = baseline_time / t
                efficiency = speedup / p
                print(f"   p={p}: {speedup:.2f}x (eficiencia: {efficiency:.2f})")

if __name__ == "__main__":
    analyze_validation()
