# final_v8_consistency_check

## 1) Run coverage
- expected runs: 27
- completed runs: 27
- failed runs: 0
- missing runs: 0
- valid_for_paper counts:
  - resource_moderate / baseline_rl: 3
  - resource_moderate / full_outer_loop: 3
  - resource_moderate / single_shot_llm: 3
  - standard_moderate / baseline_rl: 3
  - standard_moderate / full_outer_loop: 3
  - standard_moderate / single_shot_llm: 3
  - standard_severe / baseline_rl: 3
  - standard_severe / full_outer_loop: 3
  - standard_severe / single_shot_llm: 3

## 2) Method coverage
- standard_moderate: baseline_rl=yes, single_shot_llm=yes, full_outer_loop=yes, ablation_fixed_global=no
- resource_moderate: baseline_rl=yes, single_shot_llm=yes, full_outer_loop=yes, ablation_fixed_global=no
- standard_severe: baseline_rl=yes, single_shot_llm=yes, full_outer_loop=yes, ablation_fixed_global=no

## 3) Process coverage
- resource_moderate / cumulative / baseline_rl: 40 rows
- resource_moderate / cumulative / full_outer_loop: 40 rows
- resource_moderate / cumulative / single_shot_llm: 40 rows
- resource_moderate / stepwise / baseline_rl: 40 rows
- resource_moderate / stepwise / full_outer_loop: 40 rows
- resource_moderate / stepwise / single_shot_llm: 40 rows
- standard_moderate / cumulative / baseline_rl: 40 rows
- standard_moderate / cumulative / full_outer_loop: 40 rows
- standard_moderate / cumulative / single_shot_llm: 40 rows
- standard_moderate / stepwise / baseline_rl: 40 rows
- standard_moderate / stepwise / full_outer_loop: 40 rows
- standard_moderate / stepwise / single_shot_llm: 40 rows
- standard_severe / cumulative / baseline_rl: 40 rows
- standard_severe / cumulative / full_outer_loop: 40 rows
- standard_severe / cumulative / single_shot_llm: 40 rows
- standard_severe / stepwise / baseline_rl: 40 rows
- standard_severe / stepwise / full_outer_loop: 40 rows
- standard_severe / stepwise / single_shot_llm: 40 rows

## 4) Mechanism coverage
- standard_moderate / baseline_rl: action_rows=5, stage_rows=2
- standard_moderate / single_shot_llm: action_rows=6, stage_rows=2
- standard_moderate / full_outer_loop: action_rows=6, stage_rows=2
- resource_moderate / baseline_rl: action_rows=4, stage_rows=1
- resource_moderate / single_shot_llm: action_rows=5, stage_rows=1
- resource_moderate / full_outer_loop: action_rows=3, stage_rows=1
- standard_severe / baseline_rl: action_rows=5, stage_rows=2
- standard_severe / single_shot_llm: action_rows=5, stage_rows=2
- standard_severe / full_outer_loop: action_rows=5, stage_rows=2

## 5) Summary consistency
- per-seed means reproduce summary means: yes
- max absolute difference: 0.000000000000

## 6) Composition checks
- max deviation from 1 for action-category shares: 0.223611111111
- max deviation from 1 for stage shares: 0.000000000000

## 7) Data hygiene
- all CSV readable: yes
- header-only required files: none
- sentinel values present: no explicit sentinel detected
- missing required methods in plotting files: none
- stale V1/V2/V3/V4 mixed into V8 package: no

## 8) Claim support
- Full Loop improves over Baseline in high-pressure scenarios: supported
- Full Loop approaches or matches Single-shot in high-pressure scenarios: supported
- Full Loop universally dominates Single-shot across all scenarios: unsupported
- Full Loop provides traceable candidate selection and safety-validation diagnostics: supported

## Usability verdict
- package usable as candidate paper source of truth: yes