#!/bin/bash

echo "Running nvprof test for image: $1, output: $2"

for i in {1..9}
do
    for j in {1..9}
    do
        for k in {0..2}
        do
            echo "$i $j"
            echo "\"n_log\",\"$1\",\"$i\",\"$j\"" >> $2
            (nvprof --csv --normalized-time-unit ms ./build/cublurhash $i $j $1) >> $2 2>&1
        done
    done
done
