#!/bin/bash
source /home/ola/library/anaconda3/etc/profile.d/conda.sh
cd ..
python setup.py bdist_wheel
cd shell-commands
conda deactivate 
conda activate bl
pip install -e .. --upgrade
conda deactivate
conda activate py38
pip install -e .. --upgrade
