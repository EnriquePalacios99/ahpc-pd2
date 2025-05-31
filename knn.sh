#!/bin/bash
#SBATCH --job-name=knn             # Nombre del trabajo
#SBATCH --output=knn_N032_%j.log            # Archivo de salida (%j = job ID)
#SBATCH --error=error_%j.log              # Archivo de error
#SBATCH --time=00:10:00                   # Tiempo máximo de ejecución (HH:MM:SS)
#SBATCH --nodes=1                         # Número de nodos
#SBATCH --ntasks=2			  # Número total de tareas
#SBATCH --cpus-per-task=1                # Núcleos por tarea

# Cargar modulos si es necesario
ml load python3
source venv/bin/activate
module swap openmpi4 mpich/3.4.3-ofi
module load py3-mpi4py
module load py3-numpy
module load py3-scipy

# Comando principal que quieres ejecutar
mpiexec -n $SLURM_NTASKS python3.6 knn_hpc_class_vis.py 640000

module unload openmpi4 mpich/3.4.3-ofi
module unload py3-mpi4py
module unload py3-numpy
module unload py3-scipy
