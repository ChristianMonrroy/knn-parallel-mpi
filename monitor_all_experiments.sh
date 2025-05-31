#!/bin/bash

echo "🚀 Monitoreando experimentos de escalabilidad..."

while true; do
    clear
    echo "=== $(date) ==="
    
    # Trabajos en cola
    echo "📋 Trabajos en SLURM:"
    squeue -u $USER || echo "No hay trabajos en cola"
    
    echo ""
    echo "📊 Resultados generados:"
    RESULTS_COUNT=$(ls results/results_*.txt 2>/dev/null | wc -l)
    echo "Archivos de resultados: $RESULTS_COUNT/10 esperados"
    
    echo ""
    echo "📁 Archivos de resultados:"
    ls -la results/results_*.txt 2>/dev/null || echo "Ninguno aún"
    
    echo ""
    echo "📋 Últimos logs:"
    ls -t knn_digits_*.log 2>/dev/null | head -3 | while read log; do
        echo "- $log: $(tail -1 "$log" 2>/dev/null)"
    done
    
    echo ""
    if [[ $RESULTS_COUNT -ge 10 ]]; then
        echo "🎉 ¡Todos los experimentos completados!"
        break
    else
        echo "⏳ Esperando más resultados... (Ctrl+C para salir)"
    fi
    
    sleep 15
done
