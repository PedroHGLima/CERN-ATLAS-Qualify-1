#!/bin/bash
STRING=".pdf"

for file in "$PWD"/*
do
    if [[ $file == *"$STRING"* ]];then
        echo "processing $file"
        pdftocairo $file -png -r 450 -transp
    fi
done


