#!/bin/bash

echo "🚀 Ejecutando experimentos corregidos..."

# Experimentos de escalabilidad fuerte (dataset fijo mult=4)
echo "=== ESCALABILIDAD FUERTE ==="
for procs in 1 2 4 8 16; do
    echo "Enviando experimento: p=$procs, mult=4"
    sbatch --ntasks=$procs scripts/knn_digits.sh 4
    sleep 2
done

# Experimentos de escalabilidad débil (mult ≈ procs)
echo "=== ESCALABILIDAD DÉBIL ==="
configurations=(
    "1 1"
    "2 2" 
    "4 4"
    "8 8"
    "16 16"
)

for config in "${configurations[@]}"; do
    procs=$(echo $config | cut -d' ' -f1)
    mult=$(echo $config | cut -d' ' -f2)
    echo "Enviando experimento: p=$procs, mult=$mult"
    sbatch --ntasks=$procs scripts/knn_digits.sh $mult
    sleep 2
done

echo "✅ Todos los experimentos enviados"
echo "Usa 'squeue -u \$USER' para monitorear"
