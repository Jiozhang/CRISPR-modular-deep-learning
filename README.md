This branch is related to the SHAP-based feature importance calculation and result visualization.

Typically, for each dataset, 10 best-performing models (`.pt` format) are analyzed. The raw SHAP values are stored in `.csv` files prefixed with `shap_values`. The Spearman's correlation between SHAP values and feature values for each model are stored in `.csv` files prefixed with `spearman`.

The figures in the article related to SHAP analysis and the codes used to generate them can be found in `.ipynb` files prefixed with `Heatmap`.

# Installation guide
## Inside the modcrispr directory, switch to the "Feature-importance-analysis" branch
```bash
git switch Feature-importance-analysis
```
## then relaunch or refresh Jupyter Notebook

# Reference
Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. Advances in neural information processing systems **30** (2017). 
