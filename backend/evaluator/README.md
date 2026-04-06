# Retrieval Evaluator (Prototype)

This is a prototype of a generic retrieval evaluation harness, designed to work with BEIR datasets and pluggable backends.

For now, it lives inside the SmartFiles repo, but it is structured so it can be split into a separate project later.

## Layout

- `retrieval_evaluator.core` — core types and evaluator helpers.
- `retrieval_evaluator.backends` — pluggable retrieval backends (e.g., SmartFiles adapter).
- `retrieval_evaluator.logging` — run logging utilities.

## Status

This is an early implementation focused on BEIR-style evaluation with a simple backend interface.
