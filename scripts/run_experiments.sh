#!/bin/bash

# Script para ejecutar experimentos con diferentes números de procesos y tamaños de datos

# Arrays de configuraciones
PROCESSES=(1 2 4 8 16)
MULTIPLIERS=(1 2 4 8 16)

echo "Iniciando experimentos de escalabilidad KNN"
echo "Procesos: ${PROCESSES[@]}"
echo "Multiplicadores: ${MULTIPLIERS[@]}"

# Crear directorio para resultados
mkdir -p results
mkdir -p logs

# Función para ejecutar un experimento
run_experiment() {
    local procs=$1
    local mult=$2
    local job_name="knn_p${procs}_m${mult}"
    
    echo "Enviando trabajo: $job_name"
    
    # Crear script temporal
    cat > temp_${job_name}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}_%j.log
#SBATCH --error=logs/error_${job_name}_%j.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=${procs}
#SBATCH --cpus-per-task=1

# Cargar módulos
ml load python3
source venv/bin/activate
module swap openmpi4 mpich/3.4.3-ofi
module load py3-mpi4py
module load py3-numpy
module load py3-scipy
module load py3-sklearn
module load py3-matplotlib

echo "=== Iniciando experimento ==="
echo "Procesos: ${procs}"
echo "Multiplicador: ${mult}"
echo "Tiempo: \$(date)"

# Ejecutar
mpiexec -n ${procs} python3.6 knn_digits_parallel.py ${mult}

# Mover resultados
mv results_p${procs}_mult${mult}.txt results/ 2>/dev/null || true
mv predictions_p${procs}.png results/ 2>/dev/null || true

echo "=== Experimento completado ==="

# Limpiar módulos
module unload py3-matplotlib
module unload py3-sklearn
module unload py3-scipy
module unload py3-numpy
module unload py3-mpi4py
module swap mpich/3.4.3-ofi openmpi4
EOF

    # Enviar trabajo
    sbatch temp_${job_name}.sh
    
    # Limpiar script temporal
    rm temp_${job_name}.sh
    
    # Esperar un poco entre envíos
    sleep 2
}

# Experimentos de escalabilidad fuerte (mismo dataset, más procesos)
echo "=== ESCALABILIDAD FUERTE ==="
for mult in 4; do  # Dataset fijo mediano
    for procs in "${PROCESSES[@]}"; do
        run_experiment $procs $mult
    done
done

# Esperar un poco
sleep 10

# Experimentos de escalabilidad débil (más datos y más procesos proporcionalmente)
echo "=== ESCALABILIDAD DÉBIL ==="
for i in "${!PROCESSES[@]}"; do
    procs=${PROCESSES[$i]}
    mult=${MULTIPLIERS[$i]}
    run_experiment $procs $mult
done

echo "Todos los trabajos enviados. Usa 'squeue' para monitorear el progreso."
echo "Los resultados se guardarán en el directorio 'results/'"