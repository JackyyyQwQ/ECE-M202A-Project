# Device firmware (Zephyr)

Our Zephyr application for the Nano 33 BLE Sense, developed on `native_sim` first.
Status: step 2. `app/src/pcm_ring.[ch]` is the PCM ring buffer (absolute 64-bit
sample indices, all-or-nothing requests, chunked transfers that fail with
`OVERWRITTEN` if a slow link lets the audio expire). It makes no Zephyr calls.
`app/src/main.c` currently runs a self-test on a synthetic signal (sample i has
value i mod 65536) and prints `P13 RING TEST PASS`. Replay of real audio, the
request protocol and the laptop link come next.

```text
firmware/
  Dockerfile   p13-firmware image: course image + TensorFlow Lite Micro module
  run          launcher (setup, sim, sim-clean)
  app/         Zephyr application (CMakeLists.txt, prj.conf, src/)
```

## Prerequisites

- OrbStack or Docker Desktop running.
- The course base image `m202a-zephyr:4.4.2`: in the
  [course Zephyr folder](https://github.com/UCLA-ECEM202A/software/tree/83a3ac6ccf585b2b647d3d6c654b616de8c29cb8/zephyr),
  run `./zephyr setup` once. It provides Zephyr 4.4.2, west 1.5.0 and the SDK.

## Commands (from the repository root)

```bash
bash firmware/run setup
bash firmware/run sim
```

`setup` builds the `p13-firmware` image once; it only adds the TFLM module, so it
is quick. `sim` builds `app/` for `native_sim/native/64` and runs it; expect
`P13 RING TEST PASS`. `sim-clean` forces a pristine build. Build files live in the
Docker volume `p13-firmware-build`, not in the repository. Runs stop after
`SIM_TIMEOUT` seconds (default 30).

`native_sim` time is simulated: it does not measure Nano execution speed, BLE
throughput or energy. Nano builds and flashing will be added with the board step;
until then the course launcher (`./zephyr build-usb APP`) remains the reference.

## Earlier course checks

On Junchi's machine, `audio-lab/run` wraps the course launcher with a TFLM-extended
image; the course smoke and queue demos ran in `native_sim` and the course
acoustic example cross-compiled. These are course examples, not project results.
