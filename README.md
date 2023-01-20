# 2ABT Behavior Models

This project contains analysis tools and models of mouse behavior in a two-armed bandit task. The contents of this repo can be used to:

1. Characterize choice and trial-to-trial switching behavior in a 2ABT
2. Model behavior and predict from experimental data
3. Use models to simulate behavior

# How to use

### Characterize behavior

Code for visualizing choice and switching behavior of animals around block transitions in a dynamic two-armed bandit task as well as for computing and plotting conditional probabilities of behavior given action and outcome history.

### Model experimental data
Included models:
1. Hidden Markov model (HMM)
2. Logistic regression
3. Recursively formulated logistic regression (RFLR)
4. forgetting Q-learning model (FQ model)
5. sticky implementation of HMM

Supported action policies:
- Greedy
- "Stochastic"
- Softmax

The notebook `demo_models.ipynb` demonstrates how to fit and compute choice probabilities using the various models. Mouse data analyzed in Beron et al., 2022 can be found at https://doi.org/10.7910/DVN/7E0NM5 (note, this has changed from previous location). 

### Simulate behavior from models
This repo currently includes generative simulations for the HMM and RFLR.

# Installation

```
git clone https://github.com/celiaberon/2ABT_models
cd 2ABT_models
conda create -n 2abt-models python=3.8
conda activate 2abt-models
pip install -r requirements.txt
```
After setting up the virtual environment, install the SSM package for building and using the HMM following the instructions at https://github.com/lindermanlab/ssm.
