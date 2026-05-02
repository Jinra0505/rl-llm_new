# Action Mapping Note

Confirmed from ProjectRecoveryEnv action definitions:
- 0: road_A
- 1: road_B
- 2: road_C
- 3: power_A
- 4: power_B
- 5: power_C
- 6: comm_A
- 7: comm_B
- 8: comm_C
- 9: mes_to_A
- 10: mes_to_B
- 11: mes_to_C
- 12: feeder_reconfigure
- 13: coordinated_balanced
- 14: wait_hold

Action index -> action_category:
- road: 0,1,2
- power: 3,4,5
- comm: 6,7,8
- mes: 9,10,11
- feeder: 12
- coordinated: 13
- wait: 14

Mapping repair needed: yes (prior downstream mapping mismatch was repaired to road/power/comm true order).

rule_based_greedy logic:
- Uses same ProjectRecoveryEnv and action space.
- Uses shared feasibility mask `_valid_action_mask` and scores only feasible actions.
- Prefers one-step deepcopy lookahead scoring; falls back to action-type proxy only when clone/sim fails.
- Keeps wait_hold conservative; selected only when non-wait feasible actions are weak.

Trace action_category integrity:
- run_rule_based_greedy writes non-empty action_category values in traces.
