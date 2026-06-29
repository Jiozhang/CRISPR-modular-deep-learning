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
## Step 6: Download the most updated NUPACK package (current Version 4.0.2.0) from https://www.nupack.org/, and unzip the folder in /Downloads
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

### Individually trained models
<img width="1992" height="660" alt="image" src="https://github.com/user-attachments/assets/868ad57c-c0a2-45b9-bb45-866ce13ab511" />

### Fine-tuned models
<img width="2002" height="1074" alt="image" src="https://github.com/user-attachments/assets/423f79ba-02f7-4aa9-a654-8d2891d72942" />

### Enter guide sequence (in RNA) and target sequence (in DNA), follow the instructions above the input box
Note that both guide and target sequences (and template sequence if predicting prime editing reactions) should correspond to these of the protospacer strand.

<img width="1996" height="700" alt="image" src="https://github.com/user-attachments/assets/026a2a3e-e973-4ab0-bb46-ec29c8c41dcf" />

### Click on "Predict", the predicted activity will appear below
<img width="1992" height="266" alt="image" src="https://github.com/user-attachments/assets/b0024d8a-d0ef-403d-a1bf-173ed0cf92bf" />

## Custom fine-tuning
### Select reaction mode from "Cas9", "Cas12" and "Cas13"
If the spacer is located upstream of the scaffold in the guide sequence (from 5′ to 3′), it is classified as the “Cas9” category. Otherwise, if the spacer is downstream of the scaffold, the system is classified as “Cas12” when the target is DNA, or “Cas13” when the target is RNA.

<img width="2000" height="272" alt="image" src="https://github.com/user-attachments/assets/42b58a8c-cf8d-4237-b87a-767aa28412d8" />

### Provide a reaction name
<img width="1984" height="172" alt="image" src="https://github.com/user-attachments/assets/625d78e0-4fd4-444e-a78b-db381f2c5804" />

### Provide guide scaffold sequence and spacer length (maximum 30 nt)
<img width="2002" height="470" alt="image" src="https://github.com/user-attachments/assets/13205149-9be3-4479-bf45-a48c3d3f252e" />

### Select the number of times your provided dataset will be repeated to expand its size (options: 1, 10, or 100)
If your dataset is moderately large, it is not necessary to expand it during training, and you can set the repeat_k value to 1. For relatively smaller datasets (e.g., 100–1000 samples), a repeat_k value of 10 is recommended. For very small datasets (e.g., fewer than 100 samples), it is advised to use a repeat_k value of 100.

<img width="1994" height="246" alt="image" src="https://github.com/user-attachments/assets/96b5bf49-1534-405a-94db-58ab108d6c8a" />

### Enter the number of trials you wish to train the model
For improved model accuracy, it is recommended to use a larger number of trials (e.g., more than 100). However, 10-20 trials may still produce a moderately accurate model within a shorter training time.  

<img width="1986" height="234" alt="image" src="https://github.com/user-attachments/assets/aed745b6-8cf6-438c-be07-bc647c46214d" />

### Download the Excel template and upload your dataset
It should be noted that the first, second, and third columns must strictly correspond to the guide sequences, target sequences, and activity values, respectively. Both the guide and target sequences should represent the protospacer strand, with a maximum length of 30 nt. The activity values must be preprocessed (as defined by the user) to fall within the range of 0 to 1.

<img width="1994" height="332" alt="image" src="https://github.com/user-attachments/assets/f7be3c16-9633-4b4d-96f3-3a35e7ef0dda" />

### Click on "Fine-tune custom model", it will show "Training in progress......", and the progress can be monitored in the terminal
<img width="1992" height="146" alt="image" src="https://github.com/user-attachments/assets/857ab179-8625-4179-b1ce-27e563eaf8db" />

### After training is complete, the webpage will display a result summary
The MSE on both validation data and unseen data will be included. The fine-tuned model is also available for download from the result summary box.

<img width="1990" height="1012" alt="image" src="https://github.com/user-attachments/assets/595c314d-a741-4d06-a2da-de9ccbd3a2ed" />

### To predict the activity of a new guide/target pair using the fine-tuned model, enter the guide sequence (RNA) and target sequence (DNA)
Make sure the lengths of input guide and target sequences are consistent with those in the provided dataset.

<img width="1994" height="624" alt="image" src="https://github.com/user-attachments/assets/15139f36-dc51-4fc1-a905-8a32fd2b1986" />

### Click on "Predict with custom fine-tuned model", the predicted activity will appear below

<img width="1996" height="260" alt="image" src="https://github.com/user-attachments/assets/040bfdef-cb85-4d1c-86f6-a975d3dc0681" />

## The installation guide for this UI is detailed in the README.md of "Online-tool-deploy" branch

## Bug Notice
If you encounter "AttributeError: Can't get attribute 'CNN_GNN_MLP_Fusion' on <module '__main__'>" when executing "Pytorch models-Single branches.ipynb" files for loading the best single-branch models already existing in the folder, just simply replace the class name "SINGLE_BRANCH_Only" (i.e., CNN1_Only, CNN2_Only, MLP_Only, GNN_Only) with "CNN_GNN_MLP_Fusion". The error happens because the original "best_SINGLE_BRANCH.pt" files were trained with class name unchanged.



