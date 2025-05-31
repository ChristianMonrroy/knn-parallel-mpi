#!/bin/bash

echo "🔧 Instalación robusta de MPI4Py..."

# Limpiar módulos
module purge

# Verificar Python del sistema
echo "🐍 Verificando Python del sistema..."
SYSTEM_PYTHON=$(which python3)
PYTHON_VERSION=$(python3 --version)
echo "Python encontrado: $SYSTEM_PYTHON"
echo "Versión: $PYTHON_VERSION"

# Verificar/cargar MPI
echo "🔍 Verificando MPI..."
if which mpiexec &>/dev/null; then
    echo "✅ MPI ya disponible: $(which mpiexec)"
else
    echo "📦 Cargando MPI..."
    module load gnu12/12.4.0
    module load openmpi4/4.1.6
fi

# Verificar compiladores
echo "🔨 Verificando compiladores..."
echo "GCC: $(gcc --version | head -1)"
echo "MPI Compiler: $(which mpicc)"

# Recrear entorno virtual
echo "🐍 Recreando entorno virtual..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Verificar headers de Python
echo "🔍 Verificando headers de Python..."
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
echo "Include path: $PYTHON_INCLUDE"

if [[ -f "$PYTHON_INCLUDE/Python.h" ]]; then
    echo "✅ Python.h encontrado"
else
    echo "❌ Python.h no encontrado"
    echo "Intentando con headers del sistema..."
    
    # Buscar Python.h en ubicaciones comunes
    for path in /usr/include/python3.*m /usr/local/include/python3.*; do
        if [[ -f "$path/Python.h" ]]; then
            echo "✅ Encontrado en: $path"
            export CPPFLAGS="-I$path $CPPFLAGS"
            break
        fi
    done
fi

# Instalar dependencias básicas
echo "📦 Instalando dependencias básicas..."
pip install --upgrade pip
pip install numpy

# Intentar instalar mpi4py con configuración específica
echo "📦 Intentando instalar mpi4py..."
export MPICC=$(which mpicc)
pip install mpi4py --verbose

# Verificar instalación
echo "✅ Verificando instalación..."
python3 -c "
try:
    from mpi4py import MPI
    print('✅ MPI4Py instalado correctamente')
    print(f'Tamaño del comunicador: {MPI.COMM_WORLD.Get_size()}')
except ImportError as e:
    print(f'❌ Error: {e}')
"

echo "🎉 Proceso completado"
