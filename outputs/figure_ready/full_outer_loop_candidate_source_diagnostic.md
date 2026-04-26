# full_outer_loop_candidate_source_diagnostic

This diagnostic is computed from committed per-seed exports only (no rerun).

- full_outer_loop rows: 9
- rows where selected_candidate_id contains "feedback" and candidate_source is "noop_fallback": 9
- Interpretation: likely source-label/provenance aliasing where selected identifier preserves round lineage but executable source resolves to fallback/noop implementation.
- Cannot prove runtime module identity from packaged per-seed table alone; raw run artifacts are not part of this export bundle.