This is a drug discovery project using graph neural networks to classify molecules as drugs. 
It uses the BACE dataset which contains 1513 compounds with binding results against human beta-secretase 1 (BACE 1). More info on the dataset can be found here: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html

I have performed EDA, created a custom dataset, investigated several different network architectures, created automated training runs with early stopping (utilising Google Colab's GPUs),
ran model training repeats and calculated their respective metrics. 

Currently all of the relevant code for creating the dataset, functions and runnning training runs can be found in graph_classification_bace_dataset.ipynb.
I have modularised the code by producing different .py files in bace_dataset_going_modular.ipynb but working with these files is slower because Google Colabs must install
PyTorch Geometric. I have also so far had trouble installing PyTorch Geometric locally. 

My results are promising, with the GIN Conv convolutional layer (utilising edge attributes) achieving AUC scores of 96% on the test set, which is higher than results in the literature. 

My next steps for the project are to investigate different pooling methods and to work with a larger, more complex dataset. 
