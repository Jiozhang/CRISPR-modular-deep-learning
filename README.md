## Last updated: 04/11/2026

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
We have also developed a user-friendly interface (UI) that enables direct, sample-to-answer prediction across a wide range of CRISPR reactions. Users can select from individually trained models or fine-tuned models for prediction, or alternatively train a customized fine-tuned model by uploading their own dataset.

<img width="1876" height="644" alt="image" src="https://github.com/user-attachments/assets/cab4d7e9-313b-4c88-acaf-afd9386ce66f" />

## Predefined model prediction
The users can select between "Individually trained models" and "Fine-tuned models"
<img width="2002" height="212" alt="image" src="https://github.com/user-attachments/assets/18433c15-b986-473f-98ad-fc0e1142e8c1" />

### Individually trained models:
<img width="1992" height="660" alt="image" src="https://github.com/user-attachments/assets/868ad57c-c0a2-45b9-bb45-866ce13ab511" />

### Fine-tuned models:
<img width="2002" height="1074" alt="image" src="https://github.com/user-attachments/assets/423f79ba-02f7-4aa9-a654-8d2891d72942" />

### Input guide sequence (in RNA) and target sequence (in DNA), follow the instructions above the input box:
Note that both guide and target sequences (and template sequence if predicting prime editing reactions) should correspond to these of the protospacer strand.

<img width="1996" height="700" alt="image" src="https://github.com/user-attachments/assets/026a2a3e-e973-4ab0-bb46-ec29c8c41dcf" />

### Then click on "Predict", the predicted activity will appear below:

<img width="1992" height="266" alt="image" src="https://github.com/user-attachments/assets/b0024d8a-d0ef-403d-a1bf-173ed0cf92bf" />


## The installation guide for this UI is detailed in the README.md of "Online-tool-deploy" branch



