#!/bin/bash

## install dependencies

conda env create -f env_auto-xfs.yml

conda activate auto-xfs
pip install sqlalchemy==2.0.36
git clone https://github.com/kaduceo/coalitional_explanation_methods.git
git clone https://github.com/jundongl/scikit-feature.git
cd  scikit-feature/
python setup.py install
cd ..
