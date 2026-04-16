# ADR-001: Hydra for Configuration Management

## Status
Accepted

## Context
The training pipeline has many configurable parameters: data paths, model hyperparameters, training settings, and infrastructure options. We need a configuration system that:
- Supports hierarchical configuration with defaults and overrides
- Enables reproducible experiments by logging the complete resolved config
- Allows CLI overrides without code changes
- Supports multi-run sweeps for hyperparameter search

## Decision
Use **Hydra** (by Facebook Research) as the configuration framework.

## Alternatives Considered

| Option | Pros | Cons |
|---|---|---|
| **argparse** | Built-in, no dependencies | Verbose, no hierarchical configs, no sweep support |
| **YAML + manual loading** | Simple | No CLI override, no composition, no experiment tracking |
| **Hydra** | Composition, CLI override, auto-logging, sweeps | Learning curve, adds dependency |
| **gin-config** | Lightweight, binds to functions | Less structured, no file composition |

## Rationale
- Hydra's **config composition** (`defaults:` list) maps naturally to our modular architecture (data, model, training configs as separate files)
- **Auto-logging** captures the full resolved config in each run's output directory, ensuring experiment reproducibility
- **CLI overrides** (`training.epochs=10`) enable rapid iteration without editing files
- **Multi-run sweeps** (`-m training.learning_rate=0.001,0.0001`) support systematic hyperparameter exploration
- Hydra is used in the VIPer team at Bosch (mentioned in JD: "Hydra maintenance"), making this a directly relevant skill demonstration

## Consequences
- All configuration is centralized in `configs/` with clear hierarchical structure
- Experiment outputs are automatically organized by timestamp under `outputs/`
- Team members can reproduce any experiment by referencing its logged config
