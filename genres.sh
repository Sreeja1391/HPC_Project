#!/bin/bash

tar -xzf python310.tar.gz
tar -xzf python_packages.tar.gz

export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

python3 ./STAT605-Project/corr_script.py STAT605-Project/data/main_dataset_Jazz.csv
