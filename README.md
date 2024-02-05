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

To use the data reduction functions, it is necessary to install a list of libraries and clone the original repositories of the papers we are referencing to.
To clone all the repositories, install GitBash (https://git-scm.com/downloads), open GitBash and, in the "Original_Repositories" folder, execute:

```bash
./clone_repos.bat
```

To install the package, execute in a terminal: 

```bash
./install.bat
```

The file *Data_Reduction_Examples.ipynb* contains an example on how to use the functions from the package.
