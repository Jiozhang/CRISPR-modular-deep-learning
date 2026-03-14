The main branch stores the following items for each tested dataset:
1. `Raw input information files, such as guide sequences, target sequences and reaction outcome`
2. `Feature calculation codes` 
3. `Feature files`
4. `TensorFlow training scripts`
5. `TensorFlow models`
6. `PyTorch training scripts`
7. `PyTorch models`

The models are trained using either TensorFlow libraries or PyTorch libraries. The TensorFlow libraries are used for training the `CNN1`, `CNN1+CNN2`, and `CNN1+CNN2+MLP` frameworks. The PyTorch libraries are used primarily for training the full `CNN1+CNN2+MLP+GNN` framework. The TensorFlow model file extension is `.h5`, and the PyTorch model file extension is `.pt`.

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
## then relaunch or refresh Jupyter Notebook
