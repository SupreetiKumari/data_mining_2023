#!/bin/bash

if [ "$1" == "C" ]; then
    ./compress_bin $2 $3
elif [ "$1" == "D" ]; then
    ./decompress_bin $2 $3
else
    echo "Invalid input"
fi
