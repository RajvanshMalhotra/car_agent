"""Abduction/counterfactual world-model experiment package.

Kept entirely separate from `cell/`, `load/`, and the rest of the tested
simulator: this package is free to depend on numpy and torch, and nothing
under `cell/` or `load/` may import anything from here or from either
library -- see CLAUDE.md and the project rule stated in the experiment
brief. Outputs live under `runs/experiment/` (gitignored); only this source
package is committed.

Research question: can a learned world model ABDUCT a hidden, path-dependent
physical state (the sulfation crystal size in `cell/aging.py`) from an
observation window well enough to answer counterfactual queries about a
different future action sequence? See
`.superpowers/sdd/2026-09-11-battery-load-and-cell/abduction-experiment-report.md`
for the full write-up.
"""
