# Rule-Based Greedy Baseline (Feasibility-Aware)

This run adds a non-learning baseline method named `rule_based_greedy` (alias: `greedy_feasible_restoration_policy`).

At each decision step, the policy:
1. Computes the feasibility mask and feasible action set `F(s_t)` using the same environment-side feasibility logic.
2. Scores feasible actions only.
3. Prefers one-step greedy lookahead when environment cloning succeeds (via deepcopy):
   - simulate one step,
   - score by improvements in critical-load, power, communication, road, min-recovery, and progress,
   - penalize resource consumption (material and MES SOC).
4. Falls back to action-type proxy scoring if cloning is unavailable for a step.
5. Uses `wait_hold` conservatively (only when non-wait feasible actions are weak).

The method does **not** use DQN training, LLM routing, LLM candidate generation, rejection screening, or fallback candidate selection.
