#!/bin/bash

Ns=(
  10
  11
  12
  13
  14
  15
)
for n in "${Ns[@]}"
do
  to_run+="--N $n;"
done

echo "$to_run" | tr ';' '\n' | xargs -P 2 -I bash -c "python .\brute_force.py --n_alias 2 --maxiter 25 --step_size 0.2 "