#!/bin/bash

# Script de configuraci�n del entorno para KNN Paralelo
# Cluster Khipu - UTEC

echo "=================================================="
echo "Configurando entorno para KNN Paralelo con MPI"
echo "=================================================="

# Funci�n para verificar si un comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Funci�n para mostrar mensajes con colores
print_status() {
    case $1 in
        "info")
            echo -e "\033[1;34m[INFO]\033[0m $2"
            ;;
        "success")
            echo -e "\033[1;32m[SUCCESS]\033[0m $2"
            ;;
        "warning")
            echo -e "\033[1;33m[WARNING]\033[0m $2"
            ;;
        "error")
            echo -e "\033[1;31m[ERROR]\033[0m $2"
            ;;
    esac
}

# 1. Verificar que estamos en el cluster
print_status "info" "Verificando entorno del cluster..."
if [[ ! -f "/etc/slurm/slurm.conf" ]] && [[ ! $(command -v squeue) ]]; then
    print_status "warning" "No se detect� SLURM. �Est�s en el cluster Khipu?"
fi

# 2. Crear directorios necesarios
print_status "info" "Creando estructura de directorios..."
mkdir -p {results,logs,figures,temp}
print_status "success" "Directorios creados: results/, logs/, figures/, temp/"

# 3. Configurar m�dulos del cluster
print_status "info" "Cargando m�dulos del cluster..."

# Limpiar m�dulos previos
if command_exists module; then
    module purge 2>/dev/null || true
    
    # Cargar m�dulos necesarios
    module load python3 2>/dev/null || print_status "warning" "No se pudo cargar python3"
    module swap openmpi4 mpich/3.4.3-ofi 2>/dev/null || print_status "warning" "No se pudo cargar MPICH"
    module load py3-mpi4py 2>/dev/null || print_status "warning" "No se pudo cargar mpi4py"
    module load py3-numpy 2>/dev/null || print_status "warning" "No se pudo cargar numpy"
    module load py3-scipy 2>/dev/null || print_status "warning" "No se pudo cargar scipy"
    module load py3-sklearn 2>/dev/null || print_status "warning" "No se pudo cargar sklearn"
    module load py3-matplotlib 2>/dev/null || print_status "warning" "No se pudo cargar matplotlib"
    
    print_status "success" "M�dulos cargados"
    
    # Mostrar m�dulos cargados
    print_status "info" "M�dulos activos:"
    module list 2>&1 | grep -E "(python|mpi|numpy|scipy|sklearn|matplotlib)" || true
else
    print_status "warning" "Comando 'module' no encontrado. �Est�s en el cluster?"
fi

# 4. Configurar entorno virtual si es necesario
if [[ ! -d "venv" ]]; then
    print_status "info" "Creando entorno virtual..."
    if command_exists python3; then
        python3 -m venv venv
        print_status "success" "Entorno virtual creado"
    else
        print_status "error" "Python3 no encontrado"
        exit 1
    fi
fi

# Activar entorno virtual
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    print_status "success" "Entorno virtual activado"
    
    # Instalar dependencias si requirements.txt existe
    if [[ -f "requirements.txt" ]]; then
        print_status "info" "Instalando dependencias Python..."
        pip install --upgrade pip
        pip install -r requirements.txt
        print_status "success" "Dependencias instaladas"
    fi
else
    print_status "warning" "No se pudo activar el entorno virtual"
fi

# 5. Verificar instalaciones
print_status "info" "Verificando instalaciones..."

# Verificar Python y MPI
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_status "success" "Python: $PYTHON_VERSION"
else
    print_status "error" "Python3 no est� disponible"
fi

if command_exists mpiexec; then
    MPI_VERSION=$(mpiexec --version 2>&1 | head -n1)
    print_status "success" "MPI: $MPI_VERSION"
else
    print_status "error" "MPI no est� disponible"
fi

# Verificar librer�as Python
python3 -c "
import sys
libraries = ['mpi4py', 'numpy', 'sklearn', 'matplotlib', 'pandas']
for lib in libraries:
    try:
        __import__(lib)
        print(f'? {lib}')
    except ImportError:
        print(f'? {lib}')
" 2>/dev/null

# 6. Hacer ejecutables los scripts
print_status "info" "Configurando permisos de archivos..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true
print_status "success" "Permisos configurados"

# 7. Verificar archivos principales
print_status "info" "Verificando archivos del proyecto..."
required_files=(
    "src/knn_digits_parallel.py"
    "scripts/knn_digits.sh"
    "scripts/run_experiments.sh"
    "analysis/analyze_results.py"
)

all_files_present=true
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_status "success" "? $file"
    else
        print_status "error" "? $file (faltante)"
        all_files_present=false
    fi
done

# 8. Crear archivo de configuraci�n local
print_status "info" "Creando configuraci�n local..."
cat > config.env << 'EOF'
# Configuraci�n del experimento KNN
export KNN_K=3
export KNN_TEST_RATIO=0.2
export DEFAULT_MULTIPLIER=1
export MAX_PROCESSES=16
export DEFAULT_TIME="00:20:00"

# Directorios
export RESULTS_DIR="results"
export LOGS_DIR="logs"
export FIGURES_DIR="figures"

# Configuraci�n del cluster
export SLURM_PARTITION="cpu"
export SLURM_QOS="normal"
EOF
print_status "success" "Archivo config.env creado"

# 9. Resumen final
echo ""
echo "=================================================="
echo "RESUMEN DE CONFIGURACI�N"
echo "=================================================="

if $all_files_present; then
    print_status "success" "Todos los archivos necesarios est�n presentes"
else
    print_status "error" "Faltan algunos archivos. Verifica el repositorio."
fi

echo ""
print_status "info" "Pr�ximos pasos:"
echo "  1. source venv/bin/activate  # Activar entorno"
echo "  2. source config.env         # Cargar configuraci�n"
echo "  3. sbatch scripts/knn_digits.sh 1  # Experimento individual"
echo "  4. ./scripts/run_experiments.sh    # Todos los experimentos"

echo ""
print_status "info" "Comandos �tiles:"
echo "  - squeue -u \$USER           # Ver trabajos en cola"
echo "  - scancel JOBID             # Cancelar trabajo"
echo "  - tail -f logs/*.log        # Seguir logs en tiempo real"

echo ""
print_status "success" "�Configuraci�n completada!"
echo "=================================================="