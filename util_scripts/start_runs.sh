#!/bin/sh
# Start multiple wandb sweep runs in parallel
for ((n=0;n<$2;n++));
do
    wandb agent $1 &
done
