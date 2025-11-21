### create virtual env
python -m venv .venv

### activate virtual env
./.venv/scripts/Activate

### install requirements
pip install requirements.txt

### download dataset, save to env variable. .env will be created automatically
python src/download_dataset.py

### run notebooks
