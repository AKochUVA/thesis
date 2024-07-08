# Overview
Repository for the M.Sc. Thesis Project at the University of Amsterdam: Symbolic Classification for Explainable Churn Prediction

Author: Alexander Koch (alexander.koch@student.uva.nl)

Supervisor: Erman Acar (e.acar@uva.nl)

This project builds on work done by Sebastian Mežnar (2021) for HVAE and
Brence et al. (2021) for ProGED. The copyright licenses can be found in the respective
folders.

In the following you can find the installation guide, the usage guide as well as links to 
the original datasets used in the thesis.

# Installation Guide
To install and test, do the following:
1. Install rust (instructions at [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install))
2. Create a new (conda) environment
3. Install dependencies with the command: `pip install -r requirements.txt`
4. To replicate the thesis results, download the datasets and save them under data/telco/ and data/kkbox/ 

# Usage Guide
1. To run the pipeline, potentially adjust configurations in the configs/test_config.json for setting for expression 
definition, expression generation and symbolic regression settings. 
2. Adjust the Fixed and Variable Settings in the src/pipeline.py script as required (for example if you want
to load and pre-process a new dataset)
3. For better evaluation of results, the notebooks/evaluation.ipynb notebook can be used



# Datasets
Original Telco dataset: https://github.com/IBM/telco-customer-churn-on-icp4d (last accessed 02/2024), this dataset
is pre-processed in the pipeline script. 

Original KKBox dataset: https://kaggle.com/competitions/kkbox-churn-prediction-challenge (last accessed 02/2024), this 
dataset needs pre-processing to be used in the pipeline script, for that run the notebooks/EDA_KKBox.ipynb file. Be 
aware this can take multiple hours and requires approx. 30Gb of RAM or batch execution.