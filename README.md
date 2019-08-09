# deep-bayesian-irl


## Generate Trajectory and Videos

First, download the pretrained models [link](https://github.com/dsbrown1331/learning-rewards-of-learners/releases/tag/mujoco), and extract under the `models` directory.

```python run_test.py --env_id BreakoutNoFrameskip-v4 --env_type atari --model_path ~/Code/learning-rewards-of-learners/learner/models/breakout/checkpoints/03600 --record_video --episode_count 1```


You can omit the last flag `--record_video`. When it is turned on, then the videos will be recorded in a videos/ directory below the current directory.

## Learning reward function using T-REX

The main file to run is: LearnAtariReward.py

Here's an example of how to run it.


```python LearnAtariReward.py --env_name breakout --reward_model_path ./learned_models/breakout_test.params --models_dir .```
