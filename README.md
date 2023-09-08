# SurveyGreenAI

-El primer método que quiero usar es el muestreo aleatorio estratificado por clases. Es muy básico y tiene ciertos inconvenientes pero ya ha sido aplicado en "Data-Centric Green AI: An Exploratory Empirical Study" con buenos resultados y quiero usarlo como caso base. Es decir, quiero partir de que esto es lo más elemental y ver en qué medida los otros métodos lo mejoran. Para aplicarlo se puede usar esta función:

from sklearn.model_selection import StratifiedShuffleSplit

-Métodos basados en Nearest Neighbours: CNN, ENN, RENN, All-kNN (métodos wrapper, solo valen para kNN)

Están en la librería imbalanced_learn, que yo al menos lo tengo preinstalado en python pero se encuentra en este enlace
https://imbalanced-learn.org/stable/introduction.html

-PH Landmarks de Bernadette Stoltz. 

https://github.com/stolzbernadette/Outlier-robust-subsampling-techniques-for-persistent-homology

-Dominating Datasets

https://github.com/Cimagroup/Experiments-Representative-datasets

https://uses0-my.sharepoint.com/:x:/g/personal/jperera_us_es/EQi_VVO1qJ9Kr2nzCTHEPyQB0N6eozIzUuXQ98gvZbedLg?e=KVcs4W