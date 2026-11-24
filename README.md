This branch hosts a user-friendly web interface that enables users to fine-tune the provided generalist model using their own datasets, with thermodynamic features computed using ViennaRNA.

# Installation guide
The installation will take ~5 min on a regular desktop.

## 1. Open the ModularCRISPROnlineToolViennaRNA folder in Visual Studio Code 
## 2. Create a new virtual environment called modcrisprUIViennaRNA with Python 3.12.4
```bash
conda create -n modcrisprUIViennaRNA python=3.12.4 -y
```
## 3. Activate the modcrisprUIViennaRNA environment
```bash
conda activate modcrisprUIViennaRNA
```
## 4. Navigate to the project directory
```bash
cd ~/path/to/your/modcrisprUIViennaRNA
```
## 5. Install the project dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
## 6. Run the app
```bash
/path/to/your/modcrisprUI/bin/python -m uvicorn fastapi_app:app --reload
```
## 7. Go to http://127.0.0.1:8000 or any suggested address on your web browser and you should be ready to use the online version of this app
