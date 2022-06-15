#!/bin/bash

echo "Running nvprof test for image: $1, output: $2"

for i in {1..9}
do
    for j in {1..9}
    do
        echo "$i $j" 
        (nvprof --csv ./build/cublurhash $i $j $1) >> $2 2>&1
    done
done
