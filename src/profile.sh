#!/bin/bash

mkdir -p ./profiles

python -m cProfile -o profiles/out.cprof test.py

pyprof2calltree -k -i profiles/out.cprof
