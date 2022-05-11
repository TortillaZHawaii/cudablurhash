#!/bin/bash

echo "Running speed test for image: $1"
echo "For each components pair from \(2,2\) to \(8,8\) run program 5 times"
echo "Logs will be saved in: $2"

for i in {2..8}
do
    for j in {2..8}
    do
        for k in {0..2}
        do
            echo "Running for ($i, $j), pair run number $k"
            ./build/cublurhash $i $j $1 $2
        done
    done
done
