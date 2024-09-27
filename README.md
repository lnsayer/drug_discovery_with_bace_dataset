This is a drug discovery project using graph neural networks to classify molecules as drugs. 
It uses the BACE dataset which contains 1513 compounds with binding results against human beta-secretase 1 (BACE 1) (thought to be involved with Alzheimer's disease). More info on the dataset can be found [here](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html).

I have performed EDA, created a custom dataset, investigated several different network architectures, created automated training runs with early stopping (utilising my desktop's GPU),
run model training repeats and calculated their respective metrics. 

Currently all of the relevant code for creating the dataset, functions and runnning training runs can be found in this [folder](https://github.com/lnsayer/drug_discovery_with_bace_dataset/tree/main/going_modular_python). 

My project write-up can be found on my [website](https://lnsayer.github.io//my-website/2024-09-25-Graph-Neural-Network-Project-Drug-Discovery-with-the-BACE-dataset/)

My results are good, with the GIN Conv convolutional layer model (utilising edge attributes) achieving AUC scores of 88.1% on the test set, which is 0.3% off the top results in the literature. 

My future work will be more data-centric, focussing on using more data (a larger dataset) to improve predictions. 
