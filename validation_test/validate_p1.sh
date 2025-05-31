#!/bin/bash
#SBATCH --job-name=validate_p1
#SBATCH --output=validate_p1.log
#SBATCH --error=validate_p1.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "🧪 VALIDACIÓN p=1"
echo "=========================="

# Cargar módulos
ml load python3
source ../venv/bin/activate
module swap openmpi4 mpich/3.4.3-ofi
module load py3-mpi4py
module load py3-numpy
module load py3-scipy

echo "Ejecutando KNN corregido..."
mpiexec -n $SLURM_NTASKS python3 ../knn_digits_parallel_cluster_CORRECTED.py 1

echo "✅ VALIDACIÓN COMPLETADA"
