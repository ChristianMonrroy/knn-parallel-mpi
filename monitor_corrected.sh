#!/bin/bash

while true; do
    clear
    echo "=== $(date) ==="
    
    JOBS=$(squeue -u $USER | grep -v JOBID | wc -l)
    echo "📋 Trabajos activos: $JOBS"
    
    if [[ $JOBS -gt 0 ]]; then
        echo "Trabajos en ejecución:"
        squeue -u $USER
    fi
    
    echo ""
    RESULTS=$(ls results/results_*.txt 2>/dev/null | wc -l)
    echo "📊 Resultados completados: $RESULTS/10"
    
    if [[ $RESULTS -gt 0 ]]; then
        echo "Archivos generados:"
        ls -la results/results_*.txt
    fi
    
    echo ""
    if [[ $JOBS -eq 0 && $RESULTS -ge 5 ]]; then
        echo "🎉 Experimentos completados! Ejecuta el análisis:"
        echo "python3 analysis/analyze_results.py"
        break
    fi
    
    sleep 10
done
