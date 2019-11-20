#!/usr/bin/env bash
module load mpich-3.2.1/gcc-4.8.5
cd ~/CS211-HPC/lab3/build
cmake ..
make
cd ~/CS211-HPC/lab3/script
./submit.sh
watch -d -n 1 squeue -u zyang147
