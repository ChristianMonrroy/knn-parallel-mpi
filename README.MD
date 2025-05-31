# KNN Paralelo con MPI - Práctica Dirigida 02

[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![MPI](https://img.shields.io/badge/MPI-mpi4py-green.svg)](https://mpi4py.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Implementación paralela del algoritmo K-Nearest Neighbors (KNN) para clasificación de dígitos usando MPI, desarrollada para el curso de Applied High Performance Computing de UTEC.

## 📁 Estructura del Proyecto

```
knn-parallel-mpi/
├── src/
│   ├── knn_digits_parallel.py     # Implementación paralela principal
│   ├── knn_digits_sequential.py   # Versión secuencial de referencia
│   └── compare_sequential_parallel.py  # Comparación de rendimiento
├── scripts/
│   ├── knn_digits.sh             # Script SLURM individual
│   ├── run_experiments.sh        # Automatización de experimentos
│   └── setup_environment.sh      # Configuración del entorno
├── analysis/
│   ├── analyze_results.py        # Análisis completo de resultados
│   └── plot_utils.py             # Utilidades de visualización
├── docs/
│   ├── INSTRUCTIONS.md           # Instrucciones detalladas
│   └── THEORY.md                 # Fundamentos teóricos
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo
```

## 🚀 Inicio Rápido

### 1. Clonar el Repositorio
```bash
git clone https://github.com/ChristianMonrroy/knn-parallel-mpi.git
cd knn-parallel-mpi
```

### 2. Configurar Entorno (en Cluster Khipu)
```bash
# Cargar módulos y configurar entorno
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### 3. Ejecutar Experimentos
```bash
# Experimento individual
sbatch --ntasks=4 scripts/knn_digits.sh 2

# Todos los experimentos de escalabilidad
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

### 4. Analizar Resultados
```bash
python3 analysis/analyze_results.py
python3 src/compare_sequential_parallel.py
```

## 📊 Experimentos Incluidos

### Escalabilidad Fuerte
- **Dataset fijo**: multiplicador = 4 (≈7,188 muestras)
- **Procesos**: 1, 2, 4, 8, 16
- **Objetivo**: Analizar speedup con recursos crecientes

### Escalabilidad Débil  
- **Datos proporcionales**: multiplicador ≈ número de procesos
- **Configuraciones**: (p=1,mult=1), (p=2,mult=2), ..., (p=16,mult=16)
- **Objetivo**: Verificar condición teórica n_tr ∼ p×log(p)

## 📈 Métricas Analizadas

- **⏱️ Tiempos**: Total, distribución, cómputo, comunicación
- **🚀 Speedup**: S(p) = T(1)/T(p)
- **⚡ Eficiencia**: E(p) = S(p)/p
- **🎯 Precisión**: Accuracy del modelo KNN
- **📐 Validación teórica**: Ajuste del modelo T(n,p) = a + (n×k)/p + b×log(p)

## 🛠️ Requisitos del Sistema

### Software Necesario
- Python 3.6+
- MPI (MPICH o OpenMPI)
- SLURM (para cluster)

### Dependencias Python
```
mpi4py>=3.0
numpy>=1.19
scikit-learn>=0.24
matplotlib>=3.3
pandas>=1.3
```

## 📋 Uso Detallado

### Experimento Individual
```bash
# Sintaxis: sbatch --ntasks=PROCESOS scripts/knn_digits.sh MULTIPLICADOR
sbatch --ntasks=8 scripts/knn_digits.sh 4

# Monitorear progreso
squeue -u $USER

# Ver resultados
cat logs/knn_digits_*.log
```

### Configuración Personalizada
```bash
# Editar parámetros en knn_digits_parallel.py
k = 3                    # Número de vecinos
test_size_ratio = 0.2    # Porcentaje para prueba

# Editar recursos en knn_digits.sh
#SBATCH --time=00:20:00  # Tiempo máximo
#SBATCH --ntasks=8       # Número de procesos
```

## 📊 Resultados Esperados

### Archivos Generados
- `results/results_p{P}_mult{M}.txt`: Datos numéricos por experimento
- `logs/knn_digits_*.log`: Logs de ejecución SLURM  
- `knn_analysis.png`: Gráficos de análisis principal
- `theoretical_comparison.png`: Validación teórica
- `sequential_vs_parallel_comparison.png`: Comparación de implementaciones

### Métricas Típicas
- **Speedup máximo**: ~4-6x con 8-16 procesos
- **Eficiencia óptima**: ~0.7-0.8 con 2-4 procesos
- **Precisión**: ~0.97 (estable en todas las configuraciones)
- **Punto óptimo**: ~4-8 procesos según tamaño de datos

## 🔬 Fundamentos Teóricos

### Complejidad del Algoritmo
```
T(n,p) = a + (n_tr × n_te × k)/p + b×log(p)
```
- **a**: Latencia base
- **Término 2**: Cómputo distribuido O(n/p)  
- **Término 3**: Comunicación O(log p)

### Número Óptimo de Procesos
```
p_opt = √(C2/C3)
```
Donde C2 y C3 son coeficientes experimentales del modelo ajustado.

## 🐛 Solución de Problemas

### Error de Módulos
```bash
module purge
source scripts/setup_environment.sh
```

### Memoria Insuficiente
```bash
# Reducir multiplicador de datos
sbatch --ntasks=4 scripts/knn_digits.sh 1
```

### Trabajos en Cola
```bash
# Verificar estado del cluster
sinfo
squeue

# Cancelar trabajos si es necesario
scancel JOBID
```

## 📚 Referencias

- Quinn, M. J. (2003). *Parallel Programming in C with MPI and OpenMP*
- [MPI for Python Documentation](https://mpi4py.readthedocs.io/)
- [Cluster Khipu UTEC](https://docs.khipu.utec.edu.pe/)

## 🤝 Contribuciones

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍🎓 Autor

**Práctica Dirigida 02 - Applied High Performance Computing**  
Universidad de Ingeniería y Tecnología (UTEC)  
Curso: CS8084 APPLIED HIGH PERFORMANCE COMPUTING
Autores: Christian Monrroy / Oscar Bueno


---

## 📊 Badge de Estado

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-pending-yellow)
![Coverage](https://img.shields.io/badge/coverage-85%25-green)