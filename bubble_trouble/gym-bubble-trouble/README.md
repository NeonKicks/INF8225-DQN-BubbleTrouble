# gym-bubble-trouble

This is a open ai's gym adaptation of the bubble trouble game found at https://github.com/stoyaneft/bubble-trouble. It is used for RL experiments. Some modifications were made.

## Setup
1. Clone this repo.
2. `$ cd gym-bubble-trouble`
3. `$ pip install -e .`
4. Move all the content of this directory on a google drive account.
5. Start a new google collab with the `collab/bubble_trouble_dqn.ipynb'.
6. Run the notebook. 

## Repo details
The bubble trouble game was adapted with `gym` environments. To use it, one must go inside the `gym_bubbletrouble/envs` directory (TODO : Fix this). If you want to try the game, run `$ python bt.py`. If you want to train it on a local machine, run `$ python dqn.py`. If you want to change the rewards for the AI agent, edit the `settings.py` file.
