#!/bin/bash

# Eliminar directorio
rm -rf "data_reduction/Original_repositories/AIX360/aix360/models/DIPVAE"

echo "#################################"
echo "torch, torchvision, and torchaudio with cuda"
echo "#################################"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

echo "#################################"
echo "install data_reduction"
echo "#################################"
pip install .
