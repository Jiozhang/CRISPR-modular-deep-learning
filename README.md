This branch hosts a user-friendly web interface for accessing individually trained and fine-tuned models that utilize thermodynamic features computed using NUPACK. 

# Installation guide
The installation will take ~5 min on a regular desktop.

## 1. Open the ModularCRISPROnlineToolNUPACK folder in Visual Studio Code 
## 2. Create a new virtual environment called modcrisprUINUPACK with Python 3.12.4
```bash
conda create -n modcrisprUINUPACK python=3.12.4 -y
```
## 3. Activate the modcrisprUINUPACK environment
```bash
conda activate modcrisprUINUPACK
```
## 4. Navigate to the project directory
```bash
cd ~/path/to/your/modcrisprUINUPACK
```
## 5. Install the project dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
## 6. Download the most updated NUPACK package (current Version 4.0.2.0) from https://www.nupack.org/, and unzip the folder in /Downloads
## 7. Install the NUPACK package
```bash
python -m pip install -U nupack -f ~/Downloads/nupack-4.0.2.0/package
```
## 8. Run the app
```bash
/path/to/your/modcrisprUI/bin/python -m uvicorn fastapi_app:app --reload
```
## 9. Go to http://127.0.0.1:8000 or any suggested address on your web browser and you should be ready to use the online version of this app
