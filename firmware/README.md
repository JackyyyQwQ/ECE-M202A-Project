# Zephyr device application — planned

No project firmware is implemented here yet. Start with bounded PCM history and
sample-index requests in native_sim, then add the encoder and a Nano input adapter.
Keep buffer/protocol logic shared; switch microphone vs paced replay at the edge.

Reference: [course Zephyr applications](https://github.com/UCLA-ECEM202A/software/tree/83a3ac6ccf585b2b647d3d6c654b616de8c29cb8/zephyr).
`queue-demo` demonstrates queueing only; `acoustic-event` demonstrates Sense Rev2
PDM capture and stops capture during cooldown. Continuous inference will need
continuous capture instead. Attribute and record modifications when importing code.

The separate local course setup passed smoke/queue simulation and cross-compiled
the acoustic USB firmware. This is not evidence of our audio system running.
Portable project build/run commands must be added here when implemented.

Local setup note: on Junchi's machine a sibling `audio-lab/` launcher wraps the
course image/cache; from this checkout, `bash ../audio-lab/run sim queue-demo` runs
the course queue demo. In the course folder itself, `bash ./zephyr setup`,
`bash ./zephyr sim queue-demo` and `bash ./zephyr build-usb acoustic-event` set up
the toolchain, run the queue demo and build Sense Rev2 microphone firmware. These
checks are not a project pipeline or a physical board run.
