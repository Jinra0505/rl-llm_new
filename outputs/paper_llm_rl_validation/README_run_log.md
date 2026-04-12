Root cause (previous version): case preset was not reliably wired into every train/eval reset, causing same-task metrics to collapse across cases.
Fix: use train_rl.run_training(env_reset_options=...) and emit case_env_snapshot/training_env_check/eval_env_check for verification.
OK: uncertain_boundary_u1 rule
OK: uncertain_boundary_u1 llm_proxy
OK: uncertain_boundary_u2 rule
OK: uncertain_boundary_u2 llm_proxy
OK: definition_shift_d1 rule
OK: definition_shift_d1 llm_proxy
OK: definition_shift_d2 rule
OK: definition_shift_d2 llm_proxy
OK: clear_control_c1 rule
OK: clear_control_c1 llm_proxy
OK: clear_control_c2 rule
OK: clear_control_c2 llm_proxy
