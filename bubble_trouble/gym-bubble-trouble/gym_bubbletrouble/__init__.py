from gym.envs.registration import register
import logging

logger = logging.getLogger(__name__)

register(
    id='BubbleTrouble-v0',
    entry_point='gym_bubbletrouble.envs:BubbleTroubleEnv',
)
