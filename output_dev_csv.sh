#!/bin/bash

# Define the directories
baseline_dir="example/results/ptb_pos/bert-base-attn-baseline"
conditional_dir="example/results/ptb_pos/bert-base-attn-conditional"

# Define the output CSV file
output_csv="attn_ptbpos_values.csv"

# Write the header to the CSV file
echo "Layer,Baseline,Conditional" > "$output_csv"

# Loop through the 12 directories
for i in {0..11}
do
  # Construct the directory names
  baseline_subdir="$baseline_dir/bert-base-attn-${i}-norm.yaml.results"
  conditional_subdir="$conditional_dir/bert-base-attn-${i}-norm.yaml.results"
  
  # Read the value from the dev.v_entropy file in the baseline directory
  if [[ -f "$baseline_subdir/dev.v_entropy" ]]; then
    baseline_value=$(cat "$baseline_subdir/dev.v_entropy")
  fi

  # Read the value from the dev.v_entropy file in the conditional directory
  if [[ -f "$conditional_subdir/dev.v_entropy" ]]; then
    conditional_value=$(cat "$conditional_subdir/dev.v_entropy")
  fi

  # Write the values to the CSV file
  echo "$((i+1)),$baseline_value,$conditional_value" >> "$output_csv"
done

echo "CSV file has been created: $output_csv"
