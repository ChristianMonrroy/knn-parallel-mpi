#!/bin/bash
#SBATCH --job-name=knn_digits      # Nombre del trabajo
#SBATCH --output=knn_digits_%j.log # Archivo de salida (%j = job ID)
#SBATCH --error=error_digits_%j.log # Archivo de error
#SBATCH --time=00:20:00            # Tiempo máximo de ejecución
#SBATCH --nodes=1                  # Número de nodos
#SBATCH --ntasks=4                 # Número total de tareas (procesos MPI)
#SBATCH --cpus-per-task=1          # Núcleos por tarea

# Obtener parámetros
DATA_MULTIPLIER=${1:-1}

echo "=== INICIANDO EXPERIMENTO KNN ==="
echo "Procesos MPI: $SLURM_NTASKS"
echo "Multiplicador de datos: $DATA_MULTIPLIER"
echo "Nodo: $SLURMD_NODENAME"
echo "Directorio de trabajo: $(pwd)"

# Cargar SOLO los módulos que existen
echo "Cargando módulos..."
module purge
module load gnu12/12.4.0
module load hwloc/2.7.2
module load openmpi4/4.1.6
module load py3-mpi4py/3.1.3
module load py3-numpy/1.19.5
module load py3-scipy/1.5.4

echo "Módulos cargados:"
module list

# Activar entorno virtual (que tiene sklearn, matplotlib, pandas)
echo "Activando entorno virtual..."
source venv/bin/activate

# Verificar que todo está disponible
echo "Verificando entorno Python..."
python3 -c "
try:
    from mpi4py import MPI
    import numpy as np
    import sklearn
    print('✅ Todas las librerías disponibles')
    print(f'NumPy: {np.__version__}, MPI size: {MPI.COMM_WORLD.Get_size()}')
except ImportError as e:
    print(f'❌ Error de import: {e}')
    exit(1)
"

# Ejecutar el programa principal
echo "Ejecutando programa principal..."
mpiexec -n $SLURM_NTASKS python3 src/knn_digits_parallel_cluster.py $DATA_MULTIPLIER

echo "=== EXPERIMENTO COMPLETADO ==="
