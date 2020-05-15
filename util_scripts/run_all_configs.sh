#!/bin/sh
# Run all config files in a directory
for file in $1/*
do
    python main.py $2 $3 --config $file
done
