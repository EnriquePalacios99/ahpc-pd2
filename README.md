# ahpc-pd2


1. Correr en terminal 
python3 -m pip install --upgrade pip

2. Correr Requirements 
pip install -r requirements.txt


# 1) Instala Open MPI y las cabeceras de desarrollo
sudo apt-get update -y
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev

# 2) (Re)instala mpi4py enlazándolo con mpicc
pip install --no-cache-dir --force-reinstall mpi4py

#Comprueba que todo quedó en el PATH:


which mpiexec      # → /usr/bin/mpiexec
mpiexec --version  # debería mostrar la versión de Open MPI

# Ejemplo dentro de /workspaces/ahpc-pd2
mpiexec --oversubscribe -n 4 python knn_hpc_class_vis.py 10000

