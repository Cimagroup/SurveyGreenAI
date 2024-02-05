import setuptools

setuptools.setup(
    name="data_reduction",
    version="1.0",
    author="Javier Perera-Lago",
    author_email= "jperera@us.es,epaluzo@uloyola.es, vtoscano@us.es",
    description="A Data Reduction Package",
    url="https://github.com/Cimagroup/vectorization-maps",
    packages=setuptools.find_packages(),
    install_requires=[
        "scikit-learn==1.3.0",
        "scipy==1.11.4",
        "cython==3.0.2",
        "GitPython==3.1.36",
        "mnist==0.2.2",
        "pandas==1.3.5",
        "pyspark==3.5.0",
        "qpsolvers==4.0.1",
        "cvxopt==1.3.2",
        "openpyxl==3.0.10",
        "tensorflow==2.15.0",
        "xport==3.6.1",
        "cvxpy==1.4.1",
        "gudhi==3.9.0",
        "numpy==1.23.5",
    ],
)