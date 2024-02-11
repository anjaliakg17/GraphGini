# GraphGini
code for paper: "GraphGini: Fostering Individual and Group Fairness in Graph Neural Networks"

## 1. Installation Setup

Please run the following commands to install necessary packages.
For more details on Pytorch Geometric please refer to install the PyTorch Geometric packages following the instructions from [here.](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)



```
conda create --name guide python==3.7.11
conda activate guide
conda install pytorch==1.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.0.1

pip install aif360==0.3.0
```


## 2. Datasets
We ran our experiments on two high-stakes datasets: credit and income and also a larger social network dataset pokec. All the data are present in the './dataset' folder. **Due to space constraints, pokec is zipped so please unzip it before use**. 


## 3. Use bellow commands to run algorithms
```bash
cd code
python GraphGini.py 
```




## 4. Baselines
We used code provided by "https://github.com/michaelweihaosong/GUIDE.git" to run all baselines
