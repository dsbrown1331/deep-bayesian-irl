#!/bin/bash

#SBATCH --job-name bdrop_traj_breakout                                        # Job name

### Logging
#SBATCH --output=/scratch/cluster/logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=dsbrown@cs.utexas.edu  # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

### Node info
#SBATCH --partition titan                         # Queue name - current options are titans and dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                        # Number of tasks per node
#SBATCH --time 50:00:00                                                     # Run time (hh:mm:ss)

#SBATCH --gres=gpu:1                                                       # Number of gpus needed
#SBATCH --mem=5G                                                         # Memory requirements
#SBATCH --cpus-per-task=8                                              # Number of cpus needed per task

./bdrop_traj_breakout.sh
