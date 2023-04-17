#!/bin/bash

touch out.txt

for i in 32 256 480 704 928 1152 1376 1600 1824 2048 2272 2496 2720 2944 3168
do
    echo $i
    ./main $i $i $i 100 >> out.txt
done
