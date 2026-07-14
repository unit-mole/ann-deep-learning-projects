@echo off
cd /d "%~dp0\.."
call .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m src.train
