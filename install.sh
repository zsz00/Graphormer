#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

conda activate graphormer

# install requirements
pip install torch==1.9.1+cu113 torchaudio -f https://download.pytorch.org/whl/cu113/torch_stable.html
# 支持 pytorch=1.10.1, cudatoolkit=11.3, pyg=2
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch  # 10.1
conda install pyg -c pyg -c conda-forge
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.1+cu113.html
pip install torch-geometric==1.7.2
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html

cd fairseq
# if fairseq submodule has not been checkouted, run:
# git submodule update --init --recursive
pip install . --use-feature=in-tree-build
python setup.py build_ext --inplace
