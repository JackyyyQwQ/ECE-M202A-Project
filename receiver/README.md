# Laptop receiver — planned

The online receiver is not implemented. It will accept timestamped features,
maintain the causal detector state, choose audio requests, and apply refinement.
Run it as a normal Python process; it does not need Zephyr. Keep the same model
and request logic when replacing a simulated device with the Nano over USB.

Offline model definitions currently live in `models/`. In particular, the support
score in the refiner is computed after retrieval and cannot serve as a free
pre-request decision signal. See [architecture](../docs/architecture.md).
