#!/bin/bash
# Usage: bash compute_fcounts_auxfeatures_spaceinvaders.sh spaceinvaders
for value in 00025 00325 00800 01450 mean map
do
    python computePolicyExpectedFeatureCountsNetwork.py --env_name $1 --num_rollouts 30 --output_id $value --pretrained_network ../pretrained_networks/auxloss/spaceinvaders_stripped_network.params
done
