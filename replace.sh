#!/bin/bash

directory="example/bert-attn-res-ln-configs/ptb_pos/baseline"

for file in "$directory"/*
do
    if [[ -f "$file" ]]; then
        sed -i 's/label_space_size: 18/label_space_size: 50/g' "$file"
    fi
done