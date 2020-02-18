## `DemoGraph.py`
Generates demonstration videos from pretrained RL agents, and plots the encoding into the latent space as well as the decoding over time. Takes one argument, which is a pretrained network.

## `DemoGraphRunner.py`
Runs `DemoGraph.py` over every file `.params` in a folder, used to generate many demo graphs at once. Takes one argument, which is the folder in which the `.params` files can be found.

## `LatentVisualizer.py`
Opens a GUI in which the user can examine samples from the latent space, generate random samples, and slide individual dimensions to see the effect on the decoded image.

## `RandomSample.py`
Takes in a folder containing `.params` files much like `DemoGraphRunner.py`, and generates random samples from the latent space, zero samples, forward dynamics rollouts, and visualizations of greatest dimensions for a given pretrained feature encoding network.
