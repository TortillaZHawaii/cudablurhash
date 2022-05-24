#!/bin/bash

echo "Running quality test for image: $1"

for i in {1..9}
do
    for j in {1..9}
    do
        echo "$i $j `./build/cublurhash $i $j $1`" 
    done
done
