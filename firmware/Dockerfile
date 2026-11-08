# P13 firmware toolchain. Extends the course image (Zephyr 4.4.2, west 1.5.0, Zephyr
# SDK, Nordic HAL) built by the course `./zephyr setup`, and adds the TensorFlow
# Lite Micro module that older course images lack (a no-op for newer ones).
FROM m202a-zephyr:4.4.2
WORKDIR /opt/workspace
RUN west config manifest.project-filter -- +tflite-micro \
    && west update --narrow -o=--depth=1 tflite-micro
WORKDIR /work
