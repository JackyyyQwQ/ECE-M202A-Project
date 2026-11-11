# TFLM size probe: loads one int8 student (src/model_<P13_MODEL>.h, written by
# scripts/export_tflite.py), allocates its tensor arena and runs one inference.
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tflm_probe)

set(P13_MODEL ds_w025 CACHE STRING "student header src/model_<name>.h")
set(P13_ARENA_KB 200 CACHE STRING "reserved tensor arena, KB")
target_compile_definitions(app PRIVATE
  "P13_MODEL_HEADER=\"model_${P13_MODEL}.h\"" P13_ARENA_KB=${P13_ARENA_KB})
target_sources(app PRIVATE src/main.cpp)
