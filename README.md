# BoilingBench-CV

`BoilingBench-CV` is a reproducible benchmark framework for computer-vision models that identify bubbles in boiling imagery. Its first release is an **internal seed benchmark**, not yet a public data release or a claim of general validation.

## Decision supported

The benchmark determines which open-source segmentation and tracking approaches provide reliable bubble measurements across distinct boiling-image domains, and whether a replacement for the current BubbleID stack is justified by measured accuracy, domain robustness, and usability.

The framework separates static instance segmentation, per-frame morphometry derived from masks, and temporal bubble dynamics. The last requires independently reviewed temporal identities and event labels before it can be benchmarked.

## Data boundary and provenance

The repository contains code, schemas, split definitions, documentation, and evaluation records only. It does **not** include experimental images, videos, annotations, weights, or collaborator-provided data.

Initial authorized local sources are expected to include:

- `BoilingBench-5_Human_annotated_boiling_images`: Labelme polygon annotations.
- `PoolBoilingDatasets`: collaborator contour annotations for FCu-H2O, PCu-H2O, PSi-HFE, and SSi-HFE.

Original annotations are immutable inputs. Adapters create derived canonical COCO-style records while retaining original source path, source record, and a SHA-256 file hash. Redistribution, attribution, and collaborator review must be decided before any data-derived artifact is made public.

## Why grouped splits are mandatory

Frames from one source video and operating condition are correlated. A random image split would leak nearly duplicate visual information and inflate apparent generalization. All benchmark splits keep a complete acquisition group `(regime, source video/power condition)` in exactly one partition.

| Track | Training domain | Held-out domain | Scientific use |
| --- | --- | --- | --- |
| `pooled_grouped` | all regimes | unseen video/power groups | Within-suite robustness |
| `water_to_hfe` | FCu-H2O, PCu-H2O | PSi-HFE, SSi-HFE | Joint fluid/surface/optics/facility shift |
| `hfe_to_water` | PSi-HFE, SSi-HFE | FCu-H2O, PCu-H2O | Reverse transfer |
| `leave_regime_out` | three regimes | one regime | Stress test; not a causal fluid-only comparison |

HFE-vs-water is a deliberately difficult compound domain shift. It must not be interpreted as the causal effect of fluid alone because surface, imaging, and facility may differ simultaneously.

## Metrics

The primary static task is class-agnostic bubble instance segmentation. Every reported run must state model version, checkpoint, preprocessing, compute device, input size, runtime environment, split version, and random seed.

- COCO-style mask AP over IoU thresholds 0.50:0.05:0.95.
- Boundary F-score, with a declared pixel tolerance.
- Bubble count bias and MAE.
- Matched-instance centroid, equivalent-diameter, and projected-area error.
- Total projected vapor-area fraction error.
- Inference latency and a clean-environment installation record.

Aggregate scores are always accompanied by results stratified by regime, source-video group, and size bin. A model cannot be called generally robust from a pooled score alone.

## Quick start: index the collaborator contour dataset

Python 3.10+ is sufficient for the initial indexing step; no GPU or ML framework is required.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python scripts/build_poolboiling_coco.py --root "Z:\ned3-020_PoolBoilingDatasets" --output data\processed\poolboiling-v0.1
python scripts/validate_benchmark.py --annotations data\processed\poolboiling-v0.1\annotations.json --output data\processed\poolboiling-v0.1\validation.json
```

This creates a derived manifest and canonical instance annotations; it does not copy or alter raw source images or annotations.

## Model comparison policy

“All open-source models” is not a stable or reproducible population. Each candidate enters a versioned registry only if source, license, inference entrypoint, model/checkpoint identity, and output-to-instance-mask conversion are documented. Models are reported separately in two tracks:

1. **Frozen/off-the-shelf**: no benchmark-label access.
2. **Fixed-data transfer**: identical training groups, label budget, validation-only tuning, and declared seeds.

The first registry includes BubbleID-family models when their required checkpoint and compatible runtime can be verified. A model is never silently excluded because it is difficult to install; installation failure is a reported usability result.

## Status

- [x] Pool-boiling contour schema inspected and conversion adapter implemented.
- [x] Leakage-safe split generator and validator implemented.
- [ ] Labelme adapter and metadata reconciliation.
- [ ] Mask evaluator and frozen-model adapters.
- [ ] Temporal identity/event annotation specification.
- [ ] External data-rights review and public benchmark release decision.
