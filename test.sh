#!/bin/bash

touch out.txt

for i in 16 256 496 736 976 1216 1456 1696 1936 2176 2416 2656 2896 3136
do
    ./main $i $i $i 100 >> out.txt
done
