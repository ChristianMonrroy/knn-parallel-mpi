# Guía de Inicio Rápido

Esta guía te llevará desde cero hasta tener resultados en **menos de 10 minutos**.

## Setup Ultra-Rápido

### 1. Clonar y Configurar (2 minutos)
```bash
# En el cluster Khipu
git clone https://github.com/tu-usuario/knn-parallel-mpi.git
cd knn-parallel-mpi

# Configurar todo automáticamente
./scripts/setup_environment.sh

# Activar entorno
source venv/bin/activate
source config.env
```

### 2. Primer Experimento (1 minuto)
```bash
# Experimento rápido: 4 procesos, dataset pequeño
sbatch --ntasks=4 scripts/knn_digits.sh 1

# Ver si se envió correctamente
squeue -u $USER
```

### 3. Monitorear Progreso (30 segundos)
```bash
# Ver el log en tiempo real
tail -f logs/knn_digits_*.log

# O verificar estado
squeue -u $USER
```

### 4. Experimentos Completos (5 minutos)
```bash
# Todos los experimentos de escalabilidad
./scripts/run_experiments.sh

# Esto ejecutará ~10 experimentos en paralelo
# Tiempo estimado: 5-10 minutos dependiendo de la cola
```

### 5. Análisis de Resultados (1 minuto)
```bash
# Una vez completados los experimentos
python3 analysis/analyze_results.py

# Comparación secuencial vs paralelo
python3 src/compare_sequential_parallel.py
```

## Checklist Rápido

- [ ] Clonado el repositorio
- [ ] Ejecutado `setup_environment.sh`
- [ ] Enviado primer experimento con `sbatch`
- [ ] Verificado que el trabajo está en cola con `squeue`
- [ ] Ejecutado experimentos completos con `run_experiments.sh`
- [ ] Analizado resultados con `analyze_results.py`

## Resultados Esperados

Después de ejecutar todo, deberías tener:

### Archivos Generados
```
results/
+-- results_p1_mult1.txt         # Datos de rendimiento
+-- results_p2_mult2.txt
+-- ...
+-- knn_analysis.png             # Gráficos principales
+-- theoretical_comparison.png   # Validación teórica
+-- sequential_vs_parallel_comparison.png
```

### Métricas Clave
- **Speedup máximo**: ~4-6x con 8 procesos
- **Eficiencia**: >70% con 2-4 procesos
- **Precisión**: ~97% (estable)
- **Tiempo óptimo**: ~4-8 procesos

## Problemas Comunes

### El trabajo no se envía
```bash
# Verificar SLURM
sinfo
squeue

# Verificar permisos
chmod +x scripts/*.sh
```

### Error de módulos
```bash
# Recargar configuración
module purge
./scripts/setup_environment.sh
```

### Sin resultados
```bash
# Verificar que los trabajos terminaron
squeue -u $USER

# Ver logs de error
ls logs/error_*.log
cat logs/error_*.log
```

## Ayuda Rápida

### Comandos Esenciales
```bash
# Estado de trabajos
squeue -u $USER

# Cancelar trabajo
scancel JOBID

# Ver recursos disponibles
sinfo

# Seguir log en tiempo real
tail -f logs/knn_digits_*.log

# Listar resultados
ls -la results/
```

### Configuración Personalizada
```bash
# Cambiar número de procesos
sbatch --ntasks=8 scripts/knn_digits.sh 2

# Cambiar tiempo límite
# Editar scripts/knn_digits.sh
#SBATCH --time=00:30:00
```
