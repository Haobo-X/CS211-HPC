#!/usr/bin/env bash
cd ~/CS211-HPC/lab2/build
export CXX=/act/gcc-4.7.2/bin/g++
export CC=/act/gcc-4.7.2/bin/gcc
cmake ..
make
cd ~/CS211-HPC/lab2/script
sbatch submit.sh
watch -d -n 1 squeue
