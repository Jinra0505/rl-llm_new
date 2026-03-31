import numpy as np


def revise_state(state, info=None):
    capability = float(np.mean(state[12:15]))
    road_comm = float(np.mean(np.concatenate([state[3:6], state[6:9]])))
    material = float(state[20])
    switch_cap = float(state[21])
    return np.concatenate([state, np.array([capability, road_comm, material, switch_cap], dtype=np.float32)])


def intrinsic_reward(state, action, next_state, info=None, revised_state=None):
    reward = 0.0
    backbone_now = float(np.mean(state[12:15]))
    backbone_next = float(np.mean(next_state[12:15]))
    road_comm_now = float(np.mean(np.concatenate([state[3:6], state[6:9]])))
    road_comm_next = float(np.mean(np.concatenate([next_state[3:6], next_state[6:9]])))
    material_now = float(state[20])
    material_next = float(next_state[20])
    reward += 0.018 * max(0.0, backbone_next - backbone_now)
    reward += 0.014 * max(0.0, road_comm_next - road_comm_now)
    reward += 0.006 * max(0.0, float(np.mean(next_state[9:12]) - np.mean(state[9:12])))
    if material_next < material_now - 0.03:
        reward -= 0.003
    if action == 14 and (road_comm_now < 0.7 or backbone_now < 0.7):
        reward -= 0.002
    if action in [0, 1, 2, 6, 7, 8] and (road_comm_next > road_comm_now):
        reward += 0.004
    return float(reward)
