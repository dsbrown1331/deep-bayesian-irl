#!/bin/bash
for value in 00025 00325 00800 01450 mean map
do	
    python computePolicyExpectedFeatureCounts_gt.py --env_name $1 --num_rollouts 30 --output_id $value
done
