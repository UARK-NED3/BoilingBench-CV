# Model registry

Each row is a reproducibility record, not an endorsement.

| ID | Status | Track | Source/checkpoint | Output conversion | Result |
| --- | --- | --- | --- | --- | --- |
| `bubbleid-base` | pending runtime and checkpoint discovery | frozen | BubbleID | Detectron2 instance masks | not evaluated |
| `bubbleid-flow` | pending runtime and checkpoint discovery | frozen | BubbleID-Flow | Detectron2 instance masks | not evaluated |
| `bubbleid-droplet` | pending compatibility review | exploratory | BubbleID-Droplet | Detectron2 or classical fallback | not evaluated |

An entry becomes `evaluated` only after its exact source revision, checkpoint, environment, command, split hash, and machine-readable results are retained in the local results record.
