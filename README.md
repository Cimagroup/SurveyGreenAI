# Beta Version - Data Reduction Repository

This is the beta version of the data reduction repository created by the CIMAgroup research team at the University of Seville, Spain, for the European Project REXASI-PRO (REliable & eXplainable Swarm Intelligence for People with Reduced mObility) (HORIZON-CL4-HUMAN-01 programme under grant agreement nº101070028).

This repository reunites in a single function a list of data reduction techniques:

- SRS: Stratified Random Sampling
- PRD: ProtoDash Algorithm
- CLC: Clustering Centroids Selection
- MMS: Maxmin Selection
- DES: Distance-Entropy Selection
- PHL: Persistent Homology Landmarks Selection
- NRMD: Numerosity Reduction by Matrix Decomposition
- PSA: Principal Sample Analysis
- FES: Forgetting Events Selection

To use the data reduction functions, it is necessary to install a list of libraries and clone the original repositories of the papers we are referencing to. To clone them, it is necessary to 

### Installation in Windows

1. Install GitBash (https://git-scm.com/downloads).
2. Open a terminal and execute in "data_reduction/Original_Repositories":

```bash
./clone_repos.bat
```
3. To conclude the installation, go to the same location as setup.py and execute in a terminal: 

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
