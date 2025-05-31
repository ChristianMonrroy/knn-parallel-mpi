# ?? Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto KNN Paralelo con MPI! Esta guía te ayudará a participar de manera efectiva.

## ?? Tipos de Contribuciones

### ?? Reportar Bugs
- Usa el template de Issues para bugs
- Incluye información del sistema (cluster, versión MPI, etc.)
- Proporciona logs de error completos
- Describe pasos para reproducir el problema

### ?? Sugerir Mejoras
- Optimizaciones de rendimiento
- Nuevas métricas de análisis
- Mejor visualización de resultados
- Soporte para otros algoritmos ML

### ?? Mejorar Documentación
- Clarificar instrucciones
- Añadir ejemplos de uso
- Traducir a otros idiomas
- Mejorar comentarios en código

### ?? Contribuir Código
- Nuevas funcionalidades
- Optimizaciones
- Compatibilidad con otros clusters
- Tests automatizados

## ?? Proceso de Contribución

### 1. Fork y Clone
```bash
# Fork el repositorio en GitHub
# Luego clona tu fork
git clone https://github.com/TU-USUARIO/knn-parallel-mpi.git
cd knn-parallel-mpi

# Añade el repositorio original como upstream
git remote add upstream https://github.com/USUARIO-ORIGINAL/knn-parallel-mpi.git
```

### 2. Crear Rama de Desarrollo
```bash
# Crear rama descriptiva
git checkout -b feature/nueva-funcionalidad
# o
git checkout -b fix/corregir-bug
# o
git checkout -b docs/mejorar-readme
```

### 3. Desarrollar y Probar
```bash
# Hacer cambios
# Probar en el cluster
./scripts/setup_environment.sh
sbatch --ntasks=4 scripts/knn_digits.sh 1

# Verificar que no se rompe nada existente
python3 analysis/analyze_results.py
```

### 4. Commit y Push
```bash
# Staging de cambios
git add .

# Commit descriptivo
git commit -m "feat: añadir análisis de memoria por proceso

- Implementar medición de uso de memoria
- Añadir gráfico de memoria vs procesos
- Actualizar documentación con nueva métrica"

# Push a tu fork
git push origin feature/nueva-funcionalidad
```

### 5. Crear Pull Request
1. Ve a GitHub y crea un Pull Request
2. Usa el template proporcionado
3. Describe los cambios detalladamente
4. Referencia issues relacionados
5. Añade capturas de pantalla si aplica

## ?? Estándares de Código

### Python
```python
# Usar docstrings descriptivos
def calcular_speedup(tiempo_secuencial, tiempo_paralelo):
    """
    Calcula el speedup de una implementación paralela.
    
    Args:
        tiempo_secuencial (float): Tiempo de ejecución secuencial
        tiempo_paralelo (float): Tiempo de ejecución paralelo
        
    Returns:
        float: Factor de speedup (tiempo_seq / tiempo_par)
    """
    return tiempo_secuencial / tiempo_paralelo

# Nombres descriptivos
processes_count = 8  # ? Claro
p = 8               # ? Ambiguo

# Comentarios para lógica compleja
# Distribuir datos usando Scatterv para manejar remainder
comm.Scatterv([X_train, sendcounts, displs, MPI.DOUBLE], local_X, root=0)
```

### Scripts Bash
```bash
#!/bin/bash
# Siempre incluir shebang

# Verificar errores
set -e

# Funciones con documentación
configure_modules() {
    # Configura módulos necesarios para el cluster
    module load python3
    module load py3-mpi4py
}

# Variables en mayúsculas para constantes
readonly DEFAULT_PROCESSES=4
readonly MAX_TIME="00:30:00"
```

### Commits
Usa [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: nueva funcionalidad
fix: corrección de bug
docs: cambios en documentación
style: formato, sin cambios de lógica
refactor: refactoring de código
test: añadir o modificar tests
chore: tareas de mantenimiento
```

## ?? Testing

### Tests Manuales
```bash
# Probar con diferentes configuraciones
sbatch --ntasks=1 scripts/knn_digits.sh 1
sbatch --ntasks=4 scripts/knn_digits.sh 2
sbatch --ntasks=8 scripts/knn_digits.sh 4

# Verificar análisis
python3 analysis/analyze_results.py
```

### Checklist Antes de PR
- [ ] Código probado en cluster real
- [ ] Documentación actualizada
- [ ] Sin archivos temporales/resultados en commit
- [ ] Scripts son ejecutables (`chmod +x`)
- [ ] Logs muestran resultados esperados
- [ ] No rompe funcionalidad existente

## ?? Templates

### Issue de Bug
```markdown
**Descripción del Bug**
Descripción clara del problema.

**Reproducir**
Pasos para reproducir:
1. Ejecutar '...'
2. Ver error en '...'

**Comportamiento Esperado**
Qué debería pasar.

**Capturas/Logs**
```
[pegar logs aquí]
```

**Entorno:**
- Cluster: [Khipu/otro]
- MPI: [MPICH/OpenMPI]
- Python: [3.6/3.8/etc]
- Procesos: [1/2/4/8/16]
```

### Pull Request
```markdown
**Descripción**
Resumen de los cambios realizados.

**Tipo de Cambio**
- [ ] Bug fix
- [ ] Nueva funcionalidad
- [ ] Breaking change
- [ ] Documentación

**Pruebas Realizadas**
- [ ] Probado con 1 proceso
- [ ] Probado con múltiples procesos
- [ ] Análisis genera gráficos correctos
- [ ] No rompe funcionalidad existente

**Capturas de Pantalla**
Si aplica, añadir capturas de nuevos gráficos o output.
```

## ?? Áreas de Mejora Prioritarias

### Alto Impacto
1. **Optimización de comunicación**: Reduce overhead de MPI
2. **Soporte para datasets grandes**: Manejo eficiente de memoria
3. **Nuevas métricas**: Throughput, latencia, uso de memoria
4. **Visualizaciones interactivas**: Plotly, bokeh

### Medio Impacto
1. **Soporte otros algoritmos**: SVM, Random Forest paralelos
2. **Compatibilidad otros clusters**: AWS, Google Cloud
3. **Tests automatizados**: CI/CD con GitHub Actions
4. **Configuración flexible**: YAML/JSON para parámetros

### Mejoras Menores
1. **Logging mejorado**: Niveles de verbosidad
2. **Colores en output**: Mejor UX en terminal
3. **Progress bars**: Indicadores de progreso
4. **Cleanup automático**: Limpieza de archivos temporales

## ?? ¿Necesitas Ayuda?

### Recursos
- **Documentación MPI**: [mpi4py.readthedocs.io](https://mpi4py.readthedocs.io/)
- **SLURM**: [slurm.schedmd.com](https://slurm.schedmd.com/)
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org/)

### Contacto
- Abre un Issue para preguntas técnicas
- Discusiones en GitHub Discussions
- Email: [tu-email@utec.edu.pe]

## ?? Reconocimientos

### Contributors
- Lista automática de contributors en README.md
- Mención en releases para contribuciones significativas
- Badge de contributor en perfil de GitHub

### Hall of Fame
Contribuciones destacadas:
- ?? **Major Feature**: Nueva funcionalidad importante
- ?? **Performance**: Optimización significativa
- ?? **Documentation**: Mejora sustancial de docs

¡Gracias por contribuir al avance de HPC y Machine Learning! ??