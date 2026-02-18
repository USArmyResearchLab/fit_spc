#!/bin/bash 
option_directory="fit_configuration"
echo $option_directory
for file in ./$option_directory/options*.csv
do
    echo "$file"
    python fit_spc.py "$file" >> results.out
done
