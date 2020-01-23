#!/bin/bash
for value in 03125 #03425 03900 04875 map mean
do
    python ../../computePolicyExpectedFeatureCountsNetwork.py --env_name $1 --num_rollouts 100 --output_id $value --pretrained_network_dir /scratch/cluster/dsbrown/pretrained_networks/auxloss/ --postfix  _norecon_64.params_stripped.params --encoding_dims 64
done
