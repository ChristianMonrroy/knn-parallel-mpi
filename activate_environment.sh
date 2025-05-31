#!/bin/bash

echo "🚀 Activando entorno completo KNN Paralelo..."

# Cargar módulos base
module purge
module load gnu12/12.4.0
module load hwloc/2.7.2
module load openmpi4/4.1.6

# Cargar módulos científicos
module load py3-mpi4py/3.1.3
module load py3-numpy/1.19.5
module load py3-scipy/1.5.4

# Activar entorno virtual
source venv/bin/activate
source config.env

echo "✅ Entorno listo para experimentos!"
echo "Test rápido:"
python3 -c "from mpi4py import MPI; import numpy; print(f'MPI4Py + NumPy listos')"
