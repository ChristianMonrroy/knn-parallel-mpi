#!/bin/bash
#SBATCH --job-name=knn_digits      # Nombre del trabajo
#SBATCH --output=knn_digits_%j.log # Archivo de salida (%j = job ID)
#SBATCH --error=error_digits_%j.log # Archivo de error
#SBATCH --time=00:15:00            # Tiempo máximo de ejecución
#SBATCH --nodes=1                  # Número de nodos
#SBATCH --ntasks=4                 # Número total de tareas (procesos MPI)
#SBATCH --cpus-per-task=1          # Núcleos por tarea

# Cargar módulos necesarios
ml load python3
source venv/bin/activate
module swap openmpi4 mpich/3.4.3-ofi
module load py3-mpi4py
module load py3-numpy
module load py3-scipy
module load py3-sklearn
module load py3-matplotlib

# Parámetros
DATA_MULTIPLIER=${1:-1}  # Multiplicador de datos (default: 1)

echo "Ejecutando KNN con ${SLURM_NTASKS} procesos y multiplicador ${DATA_MULTIPLIER}"

# Ejecutar el programa principal
mpiexec -n $SLURM_NTASKS python3.6 knn_digits_parallel.py $DATA_MULTIPLIER

# Descargar módulos
module unload py3-matplotlib
module unload py3-sklearn
module unload py3-scipy
module unload py3-numpy
module unload py3-mpi4py
module swap mpich/3.4.3-ofi openmpi4

echo "Trabajo completado"