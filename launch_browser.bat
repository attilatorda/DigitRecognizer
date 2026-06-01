@echo off
cd /d "%~dp0"
python -m streamlit run scripts/mnist_browser_app.py
