#!/bin/bash

#SBATCH --exclusive
#SBATCH --partition=cpar
#SBATCH --constraint=k20
#SBATCH --time=02:00

time nvprof ./fluid_sim