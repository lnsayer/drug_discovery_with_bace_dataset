This is a drug discovery project using graph neural networks to classify molecules as drugs. 
It uses the BACE dataset which contains 1513 compounds with binding results against human beta-secretase 1 (BACE 1) (thought to be involved with Alzheimer's disease). More info on the dataset can be found here: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html.

I have performed EDA, created a custom dataset, investigated several different network architectures, created automated training runs with early stopping (utilising Google Colab's GPUs),
run model training repeats and calculated their respective metrics. 

Currently all of the relevant code for creating the dataset, functions and runnning training runs can be found in [graph_classification_bace_dataset.ipynb](https://github.com/lnsayer/personal_repo/blob/2ab5a255735464e96c031b8e286c57a925d1c9aa/drug%20discovery%20with%20BACE%20dataset/graph_classification_bace_dataset.ipynb).
I have modularised the code by producing different .py files in [bace_dataset_going_modular.ipynb](https://github.com/lnsayer/personal_repo/tree/f086e5e44b9d2ff97b7a6a8d5bde15e061fd23fc/drug%20discovery%20with%20BACE%20dataset/going_modular) but working with these files is slower because Google Colabs must install
PyTorch Geometric each time a package (dependent on it) is imported. I have also so far had trouble installing PyTorch Geometric locally. 

My results are promising, with the GIN Conv convolutional layer (utilising edge attributes) achieving AUC scores of 96% on the test set, which is higher than results in the literature. 

My next steps for the project are to investigate different pooling methods and to work with a larger, more complex dataset. 
