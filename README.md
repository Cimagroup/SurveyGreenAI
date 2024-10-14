# Beta Version - Data Reduction Repository

This is the beta version of the data reduction package created by the CIMAgroup research team at the University of Seville, Spain, for the European Project REXASI-PRO (REliable & eXplainable Swarm Intelligence for People with Reduced mObility) (HORIZON-CL4-HUMAN-01 programme under grant agreement nº101070028).

This repository reunites in a single package a list of data reduction techniques:

- SRS: Stratified Random Sampling
- PRD: ProtoDash Selection
- CLC: Clustering Centroids Selection
- MMS: Maxmin Selection
- DES: Distance-Entropy Selection
- PHL: Persistent Homology Landmarks Selection
- NRMD: Numerosity Reduction by Matrix Decomposition
- FES: Forgetting Events Selection

To use the data reduction functions, it is necessary to install a list of libraries and clone the original repositories of the papers we are referencing to. To clone them, it is necessary to 

### Installation in Windows

1. Create a virtual environment with Python >=3.9
2. Install GitBash (https://git-scm.com/downloads).
3. Open a terminal and execute in "data_reduction/Original_Repositories":

```bash
./clone_repos.bat
```
4. To conclude the installation, go to the same location as setup.py and execute in a terminal: 

```bash
./install.bat
```

### Installation in Ubuntu

1. Install GitBash (https://git-scm.com/downloads).
2. Open a terminal and execute in "data_reduction/Original_Repositories":

```bash
chmod +x ./clone_repos.sh
./clone_repos.sh
```

3. Create a virtual environment for Python version 3.9 or higher.

```bash
conda create -n name python=3.9
conda activate name
```

4. To conclude the installation, go to the same location as setup.py and execute in a terminal: 

```bash
chmod +x ./install.sh
sed -i -e 's/\r$//' install.sh
./install.sh
```

# Citation and reference
If you want to use our code for your experiments, please cite our paper as fpllows:

Perera-Lago J, Toscano-Duran V, Paluzo-Hidalgo E et al. An in-depth analysis of data reduction methods for sustainable deep learning [version 2; peer review: 2 approved]. Open Res Europe 2024, 4:101 (https://doi.org/10.12688/openreseurope.17554.2)

For further information, please contact us at: vtoscano@us.es jperera@us.es
