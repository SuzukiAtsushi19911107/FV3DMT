# FV3DMT

FV3DMT is a C++ codebase for 3-D magnetotelluric (MT) modeling and inversion workflows based on a finite-volume formulation.
It is designed for research, engineering, and practical applications, including large-scale numerical simulations and data-driven subsurface analysis.

This project is released under the CC0 1.0 license and is intended for unrestricted use, modification, and integration into other software.

The `mimalloc` branch uses the mimalloc library to achieve more efficient memory usage.
The developer recommends compiling this branch with Visual Studio, as mimalloc can be built and linked to the main program most easily in that environment.


---

## Features

- 3-D magnetotelluric (MT) modeling and inversion framework
- Finite-volume–based numerical implementation in C++
- Designed for large 3-D grids and computationally intensive workloads
- Parallel execution support via OpenMP
- Sample input data and utility tools included
- Makefile-based build system
- Compilation guides provided in both Japanese and English

---

## Repository Structure

- *.cpp, *.h  
  Core implementation (forward modeling, inversion, utilities, data handling, etc.)
- sample/  
  Example input files and sample workflows
- tools/  
  Helper tools for preprocessing and postprocessing
- optimlib/  
  Optimization library (vendored)
- makefile  
  Example build configuration
- compilationGuide(In_Japanese).pdf  
  Detailed build instructions (Japanese)
- compilationGuide(In_English).pdf  
  Detailed build instructions (English)

---

## Requirements

A typical build environment includes:

- C++ compiler with C++20 support
- Optional OpenMP support for parallel execution
- External libraries required by the build configuration  

Exact compiler flags, include paths, and library paths depend on your system.
Please refer to the compilation guides for concrete examples.

---

## Build

### Recommended

Follow one of the provided compilation guides:

- compilationGuide(In_Japanese).pdf
- compilationGuide(In_English).pdf

### Example (Makefile)

```bash
make
