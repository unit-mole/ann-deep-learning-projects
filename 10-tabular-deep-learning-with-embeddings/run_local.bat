@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r app\requirements.txt
streamlit run app\streamlit_app.py
endlocal
