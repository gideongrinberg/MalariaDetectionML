#!/bin/bash
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install fastai
pip install -r /home/runner/Malaria-Detection-ML/requirements.txt

python main.py