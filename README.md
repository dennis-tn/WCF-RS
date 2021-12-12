# WCF-RS
Weighted Counterfactual Explanation for Recommender System

1. data_preprocess_ml-100k.ipynb: data process for MovieLens-100K

## For NCF model, in the path: NCF/
1. configs.py: specify parameters
2. train.py: train the NCF model(black-box)
3. perturbTrain.py: using perturbed samples to train the copy model
4. MetricCalculate.py: calculate the shift rank for each perturbed sample
5. WeightedCF.py: generate minimum CF set for each user's top-1 recommendation
6. Retrain.py: retrain the black-box using modified data.
