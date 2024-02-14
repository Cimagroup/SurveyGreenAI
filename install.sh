#!/bin/bash

# Eliminar directorio
rm -rf "data_reduction/Original_repositories/AIX360/aix360/models/DIPVAE"

echo "#################################"
echo "ripser and faiss-cpu"
echo "#################################"
conda install -c conda-forge -c pytorch/label/nightly ripser==0.6.4 faiss-cpu

echo "#################################"
echo "torch, torchvision, and torchaudio with cuda"
echo "#################################"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

echo "#################################"
echo "install data_reduction"
echo "#################################"
pip install .