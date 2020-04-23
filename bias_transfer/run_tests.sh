#!/usr/bin/env bash

pip install -e ../../ml-utils
pip install -e ../../nnfabrik
pip install -e ../../nnvision
pip install -e ../../bias_transfer

python -m unittest tests
