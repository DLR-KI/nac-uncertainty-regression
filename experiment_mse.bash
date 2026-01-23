#!/bin/bash

# SPDX-FileCopyrightText: 2026 DLR e.V.
#
# SPDX-License-Identifier: MIT

set -e
# Define list of datasets
datasets=("wine" "abalone" "bikeshare" "obesity" "forest" "realestate" "concrete" "liver" "grid" "conductivity")
uncertainties=("nac" "ensemble" "mcdropout")

# Output CSV file
output_file="correlation_results_aleatoric.csv"

# Write header to CSV
echo "dataset,uncertainty,seed,correlation,d2" > "$output_file"

step () {
    dataset=$1
    uc=$2
    seed=$3
    output_file=$4
    echo "Running experiment on $dataset / $uc / seed $seed"
    # Run the experiment and capture output
    output=$(uv run regression_experiments.py --uncertainty-kind aleatoric --dataset-name "$dataset" --uncertainty-technique "$uc" --activation selu --seed "$seed")
    # Extract the correlation score using awk
    correlation=$(echo "$output" | awk '/^Correlation is/ { print $3 }')
    d2val=$(echo "$output" | awk '/^Final D2 Validation Score is/ { print $6 }')
    # Append to CSV
    echo "$dataset,$uc,$seed,$correlation,$d2val" >> "$output_file"
}

# export -f step  # export the function for subshells

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    for uc in "${uncertainties[@]}"; do
        for seed in 42 43 44; do
            step "$dataset" "$uc" "$seed" "$output_file"
        done
    done
done
echo "Results saved to $output_file"
