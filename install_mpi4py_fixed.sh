#!/bin/bash

echo "🔧 Instalación corregida de MPI4Py..."

# Limpiar módulos
module purge

# Cargar módulos en orden correcto (incluyendo dependencias)
echo "📦 Cargando módulos en orden correcto..."
module load gnu12/12.4.0
module load hwloc/2.7.2
module load ucx/1.15.0
module load libfabric/1.19.0
module load openmpi4/4.1.6

echo "✅ Módulos cargados:"
module list

# Verificar variables de entorno
echo "🔍 Verificando variables de entorno..."
echo "LD_LIBRARY_PATH contiene hwloc: $(echo $LD_LIBRARY_PATH | grep -o hwloc || echo 'NO')"

# Verificar que MPI funciona
echo "🔨 Verificando MPI..."
echo "MPI Compiler: $(which mpicc)"
mpicc --version | head -1

# Test rápido de MPI
echo "🧪 Test básico de MPI..."
cat > test_mpi.c << 'CEOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("MPI rank %d OK\n", rank);
    MPI_Finalize();
    return 0;
}
CEOF

if mpicc test_mpi.c -o test_mpi; then
    echo "✅ MPI compila correctamente"
    ./test_mpi
    rm -f test_mpi.c test_mpi
else
    echo "❌ MPI no compila"
    exit 1
fi

# Recrear entorno virtual
echo "🐍 Recreando entorno virtual..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
echo "📦 Instalando numpy..."
pip install --upgrade pip
pip install numpy

# Instalar mpi4py
echo "📦 Instalando mpi4py..."
export MPICC=$(which mpicc)
pip install mpi4py

# Verificar instalación
echo "✅ Verificando instalación..."
python3 -c "
try:
    from mpi4py import MPI
    print('✅ MPI4Py instalado correctamente')
    comm = MPI.COMM_WORLD
    print(f'Rank: {comm.Get_rank()}, Size: {comm.Get_size()}')
except ImportError as e:
    print(f'❌ Error: {e}')
except Exception as e:
    print(f'⚠️ Error en ejecución: {e}')
"

echo "🎉 Proceso completado"
