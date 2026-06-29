cmake_minimum_required(VERSION 3.18)
project(tdgw LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RTX 2060 is Turing = compute capability 7.5. Add more if you build elsewhere.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Main GPU solver.
add_executable(tdgw src/tdgw.cu)
target_compile_options(tdgw PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -O3 --use_fast_math -lineinfo>)
#   --expt-extended-lambda : the device lambda in thrust::transform_reduce
#   --use_fast_math        : fine here; conservation monitors flag any trouble
#   -lineinfo              : source attribution in Nsight Compute

# CUDA 13.x ships CCCL/Thrust headers that require MSVC's standard-conforming
# preprocessor; without this, nvcc + cl.exe stop with a #error (C1189).
if(MSVC)
  target_compile_options(tdgw PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

# Host-only verification of the integrator algorithm (no CUDA, builds with any C++17).
add_executable(host_check tests/host_split_step_check.cpp)

# ---------------------------------------------------------------------------
# Build:   cmake -B build && cmake --build build --config Release
# Run:     .\build\Release\tdgw.exe --selftest
#          .\build\Release\tdgw.exe --L 192 --D 12 --steps 20000 --dt 0.002
#          .\build\Release\host_check.exe
# Debug:   compute-sanitizer tdgw.exe --selftest      (race / OOB checks)
# Profile: ncu --set full tdgw.exe --L 192 --D 12 --steps 50
#          nsys profile tdgw.exe --L 192 --D 12 --steps 2000
# ---------------------------------------------------------------------------
