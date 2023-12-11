#!/bin/bash

tar -xzf python310.tar.gz
tar -xzf packages.tar.gz

export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

python3 corr_script.py $1
