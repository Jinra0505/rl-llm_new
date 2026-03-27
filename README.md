# Three-Task Recovery Recognition Framework

This repository is a clean, minimal project for **three-task situation recognition** in a coupled recovery environment.

## Project purpose
Primary goal:
1. Recognize a recovery scenario into exactly one task:
   - `critical_load_priority`
   - `restoration_capability_priority`
   - `global_efficiency_priority`

Secondary optional goal:
2. Use the recognized task to drive LLM-generated shaping functions (`revise_state`, `intrinsic_reward`) and DQN training.

## Task set (fixed)
Only these three tasks are used in this project:
- `critical_load_priority`
- `restoration_capability_priority`
- `global_efficiency_priority`

## Core files
- `task_recognition_prompt.py`: builds task-classification prompts from scenario context.
- `task_recognizer.py`: recognizes scenario -> one of the three tasks (rule + LLM).
- `mock_recovery_env.py`: coupled tri-layer recovery environment.
- `run_outer_loop.py`: optional full pipeline (recognition -> planning/codegen -> RL evaluation).
- `train_rl.py`: DQN training/evaluation on the recovery environment.
- `llm_client.py`: strict real-LLM client for formal runs.
- `prompts.py`: planning/codegen/feedback prompts.
- `config.yaml`: project configuration.
- `baseline_noop.py`: baseline shaping module.

## Run: task recognition only
Use this command to classify a scenario context into one of the three tasks:

```bash
python -c "from task_recognizer import ScenarioTaskRecognizer; rc={'env_summary':{'communication_recovery_ratio':0.45,'power_recovery_ratio':0.62,'road_recovery_ratio':0.40,'critical_load_shortfall':0.38,'material_stock':0.22},'trajectory_summary':{'mean_progress_delta':0.002,'constraint_violation_rate':0.03,'action_category_distribution':{'wait':0.35}}}; print(ScenarioTaskRecognizer().recognize_rule(rc))"
```

## Run: optional full pipeline
Formal end-to-end run (recognition + LLM function generation + DQN training):

```bash
python run_outer_loop.py --env project_recovery --llm-mode real --router-mode llm --reroute-each-round --config config.yaml
```

## Notes
- RL is intentionally secondary in this repository story.
- The repo is cleaned to focus on recognition-first workflow for your own three-task framework.
