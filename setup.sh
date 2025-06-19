#!/bin/bash
# Force Python 3.11 environment
pyenv install 3.11.7 -s
pyenv global 3.11.7

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run app
streamlit run app.py
