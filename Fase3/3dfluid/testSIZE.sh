#!/bin/bash

#SBATCH --exclusive
#SBATCH --partition=cpar
#SBATCH --constraint=k20
#SBATCH --time=02:00

# Array com os valores inteiros a serem utilizados
values=(20 120 60 80 100 60 80 100)

# Loop para iterar sobre os valores e executar o programa
for value in "${values[@]}"; do
    echo "Executando com valor: $value"
    time ./fluid_sim "$value"
done
