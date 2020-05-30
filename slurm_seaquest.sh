eval "$(conda shell.bash hook)"
conda activate deeplearning
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/scratch/cluster/dsbrown/tflogs/a2c/seaquest_rl python -m baselines.run --alg=a2c --env=SeaquestNoFrameskip-v4 --seed 0 --num_timesteps=5e7  --save_interval=1000 --num_env 9
