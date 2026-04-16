# ADR-004: Configurable Multimodal Fusion Strategy

## Status
Accepted

## Context
The biometric recognition system combines three modalities: left iris, right iris, and fingerprint. The method of combining (fusing) these modalities significantly impacts both model performance and interpretability. We need a fusion approach that:
- Is easily swappable for experimentation
- Supports different modality reliability scenarios
- Can be configured without code changes

## Decision
Implement a **Strategy Pattern** for fusion with two concrete strategies: **Concatenation** (default) and **Attention-based** fusion. The strategy is selected via Hydra config.

## Fusion Strategies

### Concatenation Fusion
```
[iris_left_feat] + [iris_right_feat] + [fingerprint_feat] → concat → projection → classifier
```
- **Simplest approach**: concatenate all feature vectors and project to shared space
- **Assumption**: all modalities are equally important
- **Best for**: baseline experiments, when modality quality is consistent

### Attention Fusion
```
[iris_left_feat] + [iris_right_feat] + [fingerprint_feat] → attention weights → weighted sum → classifier
```
- **Learned weighting**: network learns which modality to trust per sample
- **Best for**: real-world scenarios where modality quality varies (e.g., blurry iris but clear fingerprint)
- **Overhead**: ~10% more parameters than concatenation

## Alternatives Considered

| Strategy | Pros | Cons |
|---|---|---|
| **Early fusion** (pixel-level) | Captures cross-modal patterns early | Requires aligned modalities, heavy |
| **Concatenation** (mid-level) | Simple, effective baseline | Assumes equal modality importance |
| **Attention** (mid-level) | Adaptive weighting | More parameters, harder to interpret |
| **Late fusion** (score-level) | Modular, easy to debug | Loses cross-modal interactions |
| **Gated fusion** | Fine-grained control | Complex, needs more data |

## Rationale
- **Config-driven**: `fusion.strategy: concatenation` or `fusion.strategy: attention`
- **Shared encoder**: The iris encoder is shared between left and right eyes, reducing parameters and ensuring consistent feature spaces
- **Extensibility**: Adding a new fusion strategy requires only:
  1. Implementing the `nn.Module` with the same interface
  2. Adding it to `_FUSION_STRATEGIES` dict
  3. No changes to Trainer or DataLoader

## Consequences
- Default is concatenation (simplest, most explainable)
- Attention fusion available for experiments
- Future strategies (gated, cross-attention) can be added without modifying existing code
