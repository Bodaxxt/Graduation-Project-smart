#!/bin/bash
# Force Python 3.12.3 environment
pyenv install 3.12.3 -s
pyenv global 3.12.3

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run app
streamlit run app.py
