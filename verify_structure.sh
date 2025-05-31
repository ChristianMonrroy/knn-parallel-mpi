#!/bin/bash

echo "🔍 Verificando estructura del proyecto..."

required_files=(
    "src/knn_digits_parallel.py"
    "src/compare_sequential_parallel.py"
    "scripts/knn_digits.sh"
    "scripts/run_experiments.sh"
    "scripts/setup_environment.sh"
    "analysis/analyze_results.py"
    "README.md"
    "QUICK_START.md"
    "requirements.txt"
    ".gitignore"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ $file (faltante)"
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -eq 0 ]]; then
    echo ""
    echo "🎉 ¡Estructura completa! El proyecto está listo para GitHub."
    echo ""
    echo "Próximos pasos:"
    echo "1. git init"
    echo "2. git add ."
    echo "3. git commit -m 'Initial commit'"
    echo "4. git remote add origin https://github.com/ChristianMonrroy/knn-parallel-mpi.git"
    echo "5. git push -u origin main"
else
    echo ""
    echo "⚠���  Faltan ${#missing_files[@]} archivos. Completa la estructura antes de subir a GitHub."
fi
