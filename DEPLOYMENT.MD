# ?? Guía de Despliegue

Esta guía cubre el despliegue del proyecto KNN Paralelo desde GitHub a diferentes entornos HPC.

## ?? Pre-requisitos

### Entorno de Desarrollo
- Git 2.0+
- Python 3.6+
- Acceso a cluster HPC con SLURM
- Conexión SSH al cluster

### Cluster HPC
- SLURM workload manager
- MPI (MPICH o OpenMPI)
- Python 3.6+ con módulos científicos
- Sistema de módulos (Environment Modules)

## ?? Despliegue en Cluster Khipu (UTEC)

### 1. Conexión al Cluster
```bash
# Conectar por SSH
ssh tu-usuario@khipu.utec.edu.pe

# Verificar acceso a SLURM
sinfo
squeue
```

### 2. Clonar Repositorio
```bash
# En directorio home o workspace
cd ~/workspace  # o tu directorio preferido

# Clonar proyecto
git clone https://github.com/tu-usuario/knn-parallel-mpi.git
cd knn-parallel-mpi

# Verificar contenido
ls -la
```

### 3. Configuración Automática
```bash
# Hacer ejecutable el script de setup
chmod +x scripts/setup_environment.sh

# Ejecutar configuración automática
./scripts/setup_environment.sh

# Verificar que se completó correctamente
echo $?  # Debería mostrar 0
```

### 4. Verificación de Instalación
```bash
# Activar entorno
source venv/bin/activate
source config.env

# Verificar dependencias
python -c "import mpi4py, numpy, sklearn, matplotlib; print('? Todas las dependencias OK')"

# Verificar MPI
mpiexec --version
```

### 5. Test de Conectividad
```bash
# Test rápido con 2 procesos
sbatch --ntasks=2 scripts/knn_digits.sh 1

# Verificar que se envió
squeue -u $USER

# Esperar y verificar resultado
tail -f logs/knn_digits_*.log
```

## ?? Despliegue en Otros Clusters

### Cluster con PBS/Torque
Si tu cluster usa PBS en lugar de SLURM, modifica el script:

```bash
# Crear versión PBS del script
cp scripts/knn_digits.sh scripts/knn_digits.pbs

# Editar para usar directivas PBS
sed -i 's/#SBATCH/#PBS/g' scripts/knn_digits.pbs
sed -i 's/--job-name/-N/g' scripts/knn_digits.pbs
sed -i 's/--ntasks/-l nodes=1:ppn=/g' scripts/knn_digits.pbs
```

### AWS ParallelCluster
```bash
# Después de configurar ParallelCluster
pcluster ssh cluster-name

# Clonar y configurar como usual
git clone https://github.com/tu-usuario/knn-parallel-mpi.git
cd knn-parallel-mpi
./scripts/setup_environment.sh
```

### Google Cloud Platform
```bash
# En instancia de Compute Engine con MPI
git clone https://github.com/tu-usuario/knn-parallel-mpi.git
cd knn-parallel-mpi

# Instalar dependencias manualmente si no hay módulos
pip3 install --user -r requirements.txt

# Ejecutar sin SLURM
mpiexec -n 4 python3 src/knn_digits_parallel.py 1
```

## ?? Configuración de Monitoreo

### 1. Logging Avanzado
```bash
# Crear directorio de logs con timestamp
mkdir -p logs/$(date +%Y%m%d)

# Modificar script para logs detallados
export SLURM_LOG_DIR="logs/$(date +%Y%m%d)"
```

### 2. Monitoreo de Recursos
```bash
# Script para monitorear uso de recursos
cat > monitor_jobs.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
    echo ""
    sleep 30
done
EOF

chmod +x monitor_jobs.sh
./monitor_jobs.sh &
```

### 3. Alertas por Email (Opcional)
```bash
# Configurar notificaciones SLURM
sed -i 's/#SBATCH --mail-type=END,FAIL/#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user=tu-email@utec.edu.pe/' scripts/knn_digits.sh
```

## ?? Actualización del Proyecto

### Actualizar desde GitHub
```bash
# Ir al directorio del proyecto
cd knn-parallel-mpi

# Guardar cambios locales si los hay
git stash

# Actualizar desde origin
git pull origin main

# Restaurar cambios locales si es necesario
git stash pop

# Re-ejecutar configuración si hay cambios
./scripts/setup_environment.sh
```

### Rollback a Versión Anterior
```bash
# Ver releases disponibles
git tag -l

# Cambiar a release específico
git checkout v1.0.0

# O rollback al commit anterior
git checkout HEAD~1
```

## ?? Escalamiento de Recursos

### Configuración para Datasets Grandes
```bash
# Modificar límites de tiempo y memoria
sed -i 's/--time=00:20:00/--time=02:00:00/' scripts/knn_digits.sh
sed -i 's/--mem-per-cpu=1G/--mem-per-cpu=4G/' scripts/knn_digits.sh
```

### Experimentos de Gran Escala
```bash
# Crear script para múltiples nodos
cat > scripts/knn_multi_node.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=knn_large
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00

# Total: 32 procesos
mpiexec -n $SLURM_NTASKS python3 src/knn_digits_parallel.py 16
EOF
```

## ??? Troubleshooting

### Problemas Comunes

#### Error de Módulos
```bash
# Limpiar y recargar módulos
module purge
module avail | grep -i mpi
module load [módulo-mpi-disponible]
```

#### Error de Permisos
```bash
# Verificar y corregir permisos
find . -name "*.sh" -exec chmod +x {} \;
chmod -R u+rw results/ logs/
```

#### Error de Memoria
```bash
# Reducir tamaño del problema
sbatch --ntasks=4 scripts/knn_digits.sh 1  # dataset pequeño

# O aumentar memoria por proceso