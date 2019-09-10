# deep-bayesian-irl


## Generate Trajectory and Videos

First, download the pretrained models [link](https://github.com/dsbrown1331/learning-rewards-of-learners/releases/), and extract under the `models` directory.

```python run_test.py --env_id BreakoutNoFrameskip-v4 --env_type atari --model_path ~/Code/learning-rewards-of-learners/learner/models/breakout/checkpoints/03600 --record_video --episode_count 1```


You can omit the last flag `--record_video`. When it is turned on, then the videos will be recorded in a videos/ directory below the current directory.

## Learning reward function using T-REX

The main file to run is: LearnAtariReward.py
This will run mcmc over the network weights for Atari.
Here's an example of how to run it:


```python LinearFeatureMCMC.py --env_name breakout --reward_model_path ./learned_models/breakout_test.params --models_dir ~/Code/learning-rewards-of-learners/learner/models/ --num_mcmc_steps 5 --pretrained_network ../pretrained_networks/trex_icml/breakout_progress_masking.params```


To run RL with the mean reward from MCMC you just run

```OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/tflogs/breakout python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward mcmc_mean --custom_reward_path ~/Code/deep-bayesian-irl/pretrained_networks/trex_icml/breakout_progress_masking.params --mcmc_chain_path ~/Code/deep-bayesian-irl/mcmc_data/breakout_0.txt --seed 0 --num_timesteps=5e7  --save_interval=500 --num_env 9     ```
