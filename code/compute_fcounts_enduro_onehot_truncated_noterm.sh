#!/bin/bash
for value in 03125 03425 03900 04875 map mean
do
    python computePolicyExpectedFeatureCounts_onehot_truncated.py --env_name $1 --num_rollouts 100 --output_id $value --no_term
done
