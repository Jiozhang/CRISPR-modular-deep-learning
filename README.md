The code and data used in the paper "Generalizable Modular Deep Learning for Universal CRISPR Reaction Prediction", specifically developing single-branch and modular deep learning for diverse CRISPR reactions and their performance evaluation 
# Installation guide
## Step 1: Create a new virtual environment called modcrispr with Python 3.12.4
```bash
conda create -n modcrispr python=3.12.4 -y
```
## Step 2: Activate the modcrispr environment
```bash
conda activate modcrispr
```
## Step 3: Clone the modcrispr repository
```bash
git clone [https://github.com/Jiozhang/CRISPR-modular-deep-learning.git] /path/to/your/modcrispr
```
## Step 4: Navigate to the project directory
```bash
cd /path/to/your/modcrispr
```
## Step 5: Install the project dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
## Step 6: Install Jupyter (if not installed)
```bash
conda install notebook
```
## Step 7: Launch Jupyter Notebook
```bash
jupyter notebook
```
## To use a specific branch:
```bash
git switch branchname
```
