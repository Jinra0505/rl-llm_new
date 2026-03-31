import numpy as np


def revise_state(state, info=None):
    tri_means = np.array([np.mean(state[0:3]), np.mean(state[3:6]), np.mean(state[6:9])], dtype=np.float32)
    balance_gap = float(np.max(tri_means) - np.min(tri_means))
    progress = float(state[22])
    critical_shortfall = 1.0 - float(np.mean(state[9:12]))
    return np.concatenate([state, np.array([balance_gap, progress, critical_shortfall], dtype=np.float32)])


def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    reward = 0.0
    tri_now = np.array([np.mean(state[0:3]), np.mean(state[3:6]), np.mean(state[6:9])], dtype=np.float32)
    tri_next = np.array([np.mean(next_state[0:3]), np.mean(next_state[3:6]), np.mean(next_state[6:9])], dtype=np.float32)
    gap_now = float(np.max(tri_now) - np.min(tri_now))
    gap_next = float(np.max(tri_next) - np.min(tri_next))
    reward += 0.006 * max(0.0, float(np.mean(tri_next) - np.mean(tri_now)))
    reward += 0.007 * max(0.0, gap_now - gap_next)
    if next_state[22] > 0.8 and np.mean(next_state[9:12]) > np.mean(state[9:12]):
        reward += 0.008
    if action == 14 and next_state[22] < 0.8:
        reward -= 0.003
    return float(reward)
