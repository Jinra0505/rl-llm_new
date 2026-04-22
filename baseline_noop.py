import numpy as np


def revise_state(state, info=None):
    return np.asarray(state, dtype=np.float32)


def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    return 0.0


REWARD_CONTROLS = {}
