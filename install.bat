@echo off

echo #################################
echo ripser and faiss-cpu
echo #################################
call conda install -c conda-forge -c pytorch/label/nightly ripser==0.6.4 faiss-cpu

echo #################################
echo torch, torchvision and torchaudio with cuda
echo #################################
call pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

echo #################################
echo create distribution
echo #################################
call python setup.py sdist bdist_wheel

echo #################################
echo install data_reduction
echo #################################
call pip install dist/data_reduction-1.0.tar.gz