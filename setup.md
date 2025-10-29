# Setup
### Anaconda Setup
+ sudo apt upgrade
+ bash Anaconda3-2025.06-1-Linux-x86_64.sh
+ nano ~/.bashrc
    + export PATH=/home/nico/anaconda3/bin:$PATH
+ update -n base -c defaults conda

### Create Environment
+ conda create -c conda-forge -n sparse python=3.9 -y
+ conda activate sparse
+ conda install -c conda-forge graph-tool=2.45 -y
+ conda install -c "nvidia/label/cuda-11.8.0" cuda -y
+ pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
+ pip install torchdata
+ pip install -r requirements.txt
+ pip install -e .
+ g++ -O2 -std=c++11 -o sparse_diffusion/analysis/orca/orca sparse_diffusion/analysis/orca/orca.cpp

### Optional Fixes
+ pip install --force-reinstall "torch==2.2.2+cu118" "torchdata==0.7.1" --extra-index-url https://download.pytorch.org/whl/cu118
+ pip uninstall -y dgl
+ pip install dgl==1.1.2
+ pip install yacs
+ pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
+ pip uninstall networkx
+ pip install "networkx<3.0"
+ pip install fcd
+ pip install fcd-torch


### Run with:
+ conda activate sparse
+ python3 main.py
