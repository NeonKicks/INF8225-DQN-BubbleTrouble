import gym
import bt
import random
import time

# TODO : Make is we don't need to be in this directory to import bubble_trouble
# TODO : Add game states to game.py

LEFT = 0
RIGHT = 1
FIRE = 2


class BubbleTroubleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rand=False):
        self.action_space = gym.spaces.Discrete(3)
        self.state = None
        self.rand = rand
        self.seed()

    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        assert self.action_space.contains(action), '%r (%s) invalid' % (action, type(action))
        bt.game.update(restart=False)
        self.update()

        restart = bt.game.game_over or bt.game.level_completed
        self.state = self.render()

        self.take_action(action)
        reward = bt.game.reward

        return self.state, reward, restart, {}

    def reset(self):
        bt.game.restart(rand=self.rand)
        self.update()

    @staticmethod
    def update():
        bt.draw_world()
        bt.clock.tick(30)

    def render(self, mode='rgb_array', *args, **kwargs):
        assert mode == 'rgb_array'
        image = bt.get_numpy_array()
        return image.swapaxes(1, 2)

    @staticmethod
    def take_action(action):
        if action == LEFT:
            bt.game.player.moving_left = True
            bt.game.reward += bt.REWARD_MOVE # Adding reward here
        elif action == RIGHT:
            bt.game.player.moving_right = True
            bt.game.reward += bt.REWARD_MOVE
        elif action == FIRE and not bt.game.player.weapon.is_active:
            bt.game.player.shoot()

    def seed(self, seed=time.time()):
        random.seed(seed)

    def close(self):
        bt.quit_game()
