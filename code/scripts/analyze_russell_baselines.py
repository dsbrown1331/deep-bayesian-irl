import numpy as np
import os
location = "/home/dsbrown/Downloads/Russell_baseline_evals"

game_names = ['beamrider', 'breakout', 'enduro', 'seaquest', 'spaceinvaders']
print("ensemble scores")
file_name = "_ensemble_43000_evaluation.txt"
ensemble_means = {}
for g in game_names:
    full_path = os.path.join(location, g + file_name)
    #print(full_path)
    game_rollouts = []
    if os.path.exists(full_path):
        f = open(full_path)
        for line in f:
            if line.strip() is not None:
                game_rollouts.append(float(line.strip()))
        print(g, np.mean(game_rollouts), len(game_rollouts))
        ensemble_means[g] = np.mean(game_rollouts)
print()
print("dropout scores")
file_name = "_dropout_43000_evaluation.txt"
dropout_means = {}
for g in game_names:
    full_path = os.path.join(location, g + file_name)
    #print(full_path)
    game_rollouts = []
    if os.path.exists(full_path):
        f = open(full_path)
        for line in f:
            if line.strip() is not None:
                game_rollouts.append(float(line.strip()))
        print(g, np.mean(game_rollouts), len(game_rollouts))
        dropout_means[g] = np.mean(game_rollouts)
