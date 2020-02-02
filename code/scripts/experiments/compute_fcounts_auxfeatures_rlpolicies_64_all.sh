#!/bin/bash
for value in 00001 04000 08000 12000 16000 20000 24000 28000	32000 36000 40000 00500 01000 01500 02000 06000 10000 14000 18000 22000 26000 30000	34000 38000 42000 02500 03000 07000 11000 15000 19000 23000 27000 31000 35000 39000 43000
do
    python ../../computePolicyExpectedFeatureCountsNetwork.py --env_name $1 --num_rollouts 100 --output_id $value --pretrained_network_dir /scratch/cluster/dsbrown/pretrained_networks/auxloss/ --postfix  _64_all.params_stripped.params --encoding_dims 64 --fcount_dir /scratch/cluster/dsbrown/rl_policies/ --rl_eval
done
