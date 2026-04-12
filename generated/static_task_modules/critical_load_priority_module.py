import numpy as np


def revise_state(state, info=None):
    critical_total = float(state[9] + state[10] + state[11]) / 3.0
    power_mean = float(np.mean(state[0:3]))
    late_stage = float(state[22] > 0.75)
    min_critical = float(np.min(state[9:12]))
    return np.concatenate([state, np.array([critical_total, power_mean, late_stage, min_critical], dtype=np.float32)])


def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    reward = 0.0
    crit_now = float(state[9] + state[10] + state[11])
    crit_next = float(next_state[9] + next_state[10] + next_state[11])
    power_now = float(np.mean(state[0:3]))
    power_next = float(np.mean(next_state[0:3]))
    reward += 0.015 * (crit_next - crit_now)
    reward += 0.006 * max(0.0, power_next - power_now)
    if next_state[22] > 0.8 and crit_next > crit_now:
        reward += 0.01
    if action == 14 and next_state[22] < 0.8:
        reward -= 0.002
    return float(reward)
