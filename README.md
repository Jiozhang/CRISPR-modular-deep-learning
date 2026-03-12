This branch stores the datasets and codes used for transfer learning. The transfer learning strategy is used to construct an expert model for unobserved CRISPR enzymes or reaction modalities from a pre-trained generalist model covering basic CRISPR reactions. 

The "Generalist model" folder contains the feature calculation, model training and validation on trained, untrained and completely unobserved datasets regarding generalist models under both the `CNN1-only` and the full `CNN1+CNN2+MLP+GNN` frameworks. 

The "Expert model" folder contains the calculated features and the codes for zero-shot prediction using the generalist model, few-shot prediction with a fine-tuned generalist model (expert model), as well as the benchmarking model trained with the full datasets under the `CNN1+CNN2+MLP+GNN` framework for performance comparison. The tested CRISPR reactions span Cas9 variants, small Cas9 variants, Cas12a variants, base editors and prime editors. 

# Installation guide
## Inside the modcrispr directory, switch to the "Transfer-learning" branch
```bash
git switch Transfer-learning
```
## then relaunch or refresh Jupyter Notebook
