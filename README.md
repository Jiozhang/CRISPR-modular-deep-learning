The main branch stores the following items for each tested dataset:
1. `Raw input information files, such as guide sequences, target sequences and reaction outcome`
2. `Feature calculation codes` 
3. `Feature files`
4. `TensorFlow training scripts`
5. `TensorFlow models`
6. `PyTorch training scripts`
7. `PyTorch models`

The models are trained using either TensorFlow libraries or PyTorch libraries. The TensorFlow libraries are used for training the `CNN1`, `CNN1+CNN2`, and `CNN1+CNN2+MLP` frameworks. The PyTorch libraries are used primarily for training the full `CNN1+CNN2+MLP+GNN` framework. The TensorFlow model file extension is `.h5`, and the PyTorch model file extension is `.pt`.

# Installation guide (with Anaconda)
## Step 0: Install Anaconda Navigator (https://www.anaconda.com/download)
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
git clone https://github.com/Jiozhang/CRISPR-modular-deep-learning.git ~/path/to/your/modcrispr
```
## Step 4: Navigate to the project directory
```bash
cd ~/path/to/your/modcrispr
```
## Step 5: Install the project dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
## Step 6: Download the most updated NUPACK package (currently Version 4.0.2.0) from https://www.nupack.org/, and unzip the folder in /Downloads
## Step 7: Install the NUPACK package
```bash
python -m pip install -U nupack -f ~/Downloads/nupack-4.0.2.0/package
```
## Step 8: Install Jupyter (if not installed)
```bash
conda install notebook
```
## Step 9: Launch Jupyter Notebook
```bash
jupyter notebook
```
## To use a specific branch, first shutdown the current notebook server and get your prompt back:
press:
```bash
Control + C
```
then when it asks something like: 
```
Shutdown this notebook server (y/[n])?
```
type:
```bash
y
```
then in the terminal, type:
```bash
git switch branchname
```
## then relaunch or refresh Jupyter Notebook
```bash
jupyter notebook
```
# Web UI version 
We have also designed a user interface (UI) which enables direct sample-to-answer prediction of a diverse range of CRISPR reactions. The users are able to select a reaction from individually trained models or fine-tuned models. 

## Individually trained models:
<img width="2044" height="886" alt="image" src="https://github.com/user-attachments/assets/dbe0b6cf-d142-4c8b-86f7-0746540bc7ac" />

## Fine-tuned models:
<img width="2036" height="1066" alt="image" src="https://github.com/user-attachments/assets/3f60f4ab-4acb-4b6c-a7b0-d670c362a3c0" />

## Simply input guide sequence (in RNA) and target sequence (in DNA), follow the instructions above the input box:
<img width="2020" height="846" alt="image" src="https://github.com/user-attachments/assets/6d3587c3-df0d-4427-8013-5472f6acfc70" />

## Then click on "Predict", the predicted activity will appear below:
<img width="2020" height="400" alt="image" src="https://github.com/user-attachments/assets/69d517e7-1064-42c1-8668-90b2d3171ff4" />





