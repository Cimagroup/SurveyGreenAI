# Beta Version - Data Reduction Repository

This is the beta version of the data reduction repository created by the CIMAgroup research team at the University of Seville, Spain, for the European Project REXASI-PRO (REliable & eXplainable Swarm Intelligence for People with Reduced mObility) (HORIZON-CL4-HUMAN-01 programme under grant agreement nº101070028).

This repository reunites in a single function a list of data reduction techniques:

- SRS: Stratified Random Sampling
- CLC: Clustering Centroids Selection
- MMS: Maxmin Selection
- DES: Distance-Entropy Selection
- DOM: Dominating Dataset Selection
- PHL: Persistent Homology Landmarks Selection
- NRMD: Numerosity Reduction by Matrix Decomposition
- PSA: Principal Sample Analysis
- PRD: ProtoDash Algorithm

To use the data reduction functions, it is necessary to install a list of libraries and clone the original repositories of the papers we are referencing to.

To install all the required libraries, execute in a terminal:

```bash
./conda-requirements.bat
pip install -r requirements.txt
``` 
To clone all the repositories, install GitBash ([GitBash Download](https://git-scm.com/downloads)), open GitBash, and, in the "Original_Repositories" folder, execute:

```bash
./clone_repos.bat
```
The details of the data reduction algorithms can be read in "reduction_techniques.py". We recommend importing the function data_reduction from "main.py," which can perform any of the listed data reduction methods by changing the method parameter.

To run an experiment comparing all the reduction techniques, go into the "Experiments" folder and run one of the five examples we provide. The results of the experiments will be saved in the folder "Results".

