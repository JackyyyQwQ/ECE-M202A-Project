# Laptop receiver — planned

The online receiver is not implemented. It will accept timestamped features,
maintain causal detector state, and choose REPORT, bounded LOOKAHEAD via BiGRU,
or PCM REFINE. BiGRU and controller are planned; compare paths before selection.
Run it as a normal Python process; it does not need Zephyr. Keep the same model
and request logic when replacing a simulated device with the Nano over USB.

Offline model definitions currently live in `models/`. In particular, the support
score in the refiner is computed after retrieval and cannot serve as a free
pre-request decision signal. See [architecture](../docs/architecture.md).

Use only received features for bounded bidirectional inference, aligned to the
original target interval. Waiting and inference count toward the reporting deadline.
The first controller selects one action per candidate. Keep the context-only
control and compare refinement with a similar-duration lookahead.
