## A machine learning-based analysis for the proper identification of species.
---
This project aims to verify or reasses the validity of species classified by humans.
This repository contains:
- main.py: The only file you need to run
- pipe.py: Called by main.py. It contains class definitions for supervised and unsupervised learning. 
  - GetData(): Performs z-score normalization on numerical data
  - DoKFold(): Supervised learning, runs a KFold cross-validation with Decision Trees of balanced weight. See: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
  - DoClustering(): Unsupervised learning, runs KMean clustering on data

---
Requirements:
1) Takes a .csv as input
2) Requres the last two columns to be ID and species
3) Requires all other columns to be numeric.
