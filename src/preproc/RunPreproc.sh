#!/bin/bash

echo "Making dirs..."

mkdir -p ../../data/task-1/preproc
mkdir -p ../../data/task-2/preproc
mkdir -p ../../data/FunLines/task-1/preproc
mkdir -p ../../data/FunLines/task-2/preproc

echo "Copying original data to protobuf..."
python 1_saveoriginal.py

echo "Handling concatenated words..."
python 2_concat.py
