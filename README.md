This branch stores the user-friendly webpage version of individually trained models and fine-tuned models

# Installation guide

## 1. Open the ModularCRISPROnlineTool folder in Visual Studio Code 
## 2: Create a new virtual environment called modcrisprUI with Python 3.12.4
```bash
conda create -n modcrisprUI python=3.12.4 -y
```
## 3: Activate the modcrisprUI environment
```bash
conda activate modcrisprUI
```
## 4: Navigate to the project directory
```bash
cd ~/path/to/your/modcrisprUI
```
## 5: Install the project dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
## 6: Download the most updated NUPACK package (currently Version 4.0.2.0) from https://www.nupack.org/, and unzip the folder in /Downloads
## 7: Install the NUPACK package
```bash
python -m pip install -U nupack -f ~/Downloads/nupack-4.0.2.0/package
```
## 8. Run the app
```bash
/path/to/your/modcrisprUI/bin/python -m uvicorn fastapi_app:app --reload
```
## 9. Go to http://127.0.0.1:8000 or any suggested address on your web browser and you should be ready to use the online version of this app
