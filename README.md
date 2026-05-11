# BLAS Baseliner Benchmark

A cross-vendor BLAS benchmarking suite built on the Baseliner framework, targeting cuBLAS (NVIDIA) and rocBLAS (AMD) implementations. This benchmark demonstrates Baseliner's hardware-agnostic approach to GPU performance measurement through a plugin-oriented architecture.

## What is this benchmark?

`blas-baseliner` is a validation benchmark suite implementing Basic Linear Algebra Subprograms (BLAS) operations across multiple GPU vendors. Currently, it focuses on General Matrix Multiply (GEMM) operations with support for:

- **Operations**: GEMM (multiple variants: regular, batched, strided batched, extended precision, 3m)
- **Data types**: `float`, `half` precision
- **Backends**: CUDA (cuBLAS) and HIP (rocBLAS)

The suite uses a template-based architecture to maximize code reuse across operations, data types, and backends. It registers 50 distinct workloads per backend from just nine GEMM specializations, demonstrating the efficiency of the Baseliner framework.

## Dependencies

### Required

- **CMake** >= 3.15
- **C++ compiler** with C++17 support
- **At least one GPU backend**:
  - **CUDA Toolkit** >= 12.0 (for NVIDIA GPUs)
  - **ROCm** with rocBLAS (for AMD GPUs)

### Automatically Fetched

- **baseliner-adapter** (v1.0) - Automatically downloaded via CMake FetchContent from https://github.com/comeyrd/baseliner-adapter.git

### Runtime Requirements

- NVIDIA GPU (CUDA-capable) or AMD GPU (ROCm-capable)
- Appropriate GPU drivers installed

## Compilation

### Basic Build

The project uses CMake for configuration and building:

```bash
# Create and enter build directory
mkdir build && cd build

# Configure (detects available GPU compiler automatically)
cmake ..

# Build
cmake --build .
```

### Backend-Specific Builds

The build system automatically detects available GPU compilers (CUDA or HIP). To explicitly control which backend is used:

#### CUDA Only
```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

#### HIP Only
```bash
cmake -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc ..
```

### Build Configuration Options

The project sets the following defaults:
- C++ Standard: C++17
- CUDA Runtime Library: Shared
- Runtime output directory: `${CMAKE_BINARY_DIR}`

**Note**: If neither CUDA nor HIP compiler is detected, CMake will fail with an error: "Requires at least one device compiler."

## Running the Benchmark

After compilation, the benchmark executable will be located in the build directory. The specific name depends on your CMake configuration (check the `gpu-blas/` subdirectory build output).

### Basic Execution

```bash
# From the build directory
./blas-baseliner run 
```

### Protocol Files

The benchmark is controlled through JSON protocol files located in the `config/` directory:

- `config/single_shot.protocol.json` - Single execution configuration
- `config/auto_bz_variation_criterion.json` - Automatic batch size variation with convergence criteria

Example protocol file specifies:
- Workloads to run
- Stopping criteria (fixed count, entropy convergence, coefficient of variation)
- Measurement parameters (batch size, cache flushing, thermal stabilization)
- Output statistics to collect

## Computing Lines of Code

A Python script is provided to analyze the implementation effort and code organization:

```bash
python3 lines-of-code.py
```
### Output

The script produces a detailed breakdown including:

- Total source lines of code
- Cross-backend shared code vs. backend-specific code
- Per-backend statistics
- Per-shape statistics
- Specialization granularity
- Average LOC per workload

The output is saved to `lines-of-code-output.txt` and includes:
- Implementation effort metrics
- Code reuse statistics
- Workload density analysis
- Hierarchical breakdown by backend/shape/specialization

## Project Structure

```
blas-baseliner-benchmark/
├── CMakeLists.txt              # Main build configuration
├── gpu-blas/                   # BLAS implementation source
│   ├── BlasShapes.hpp          # Shape definitions (matrix dimensions, args)
│   ├── cuda/                   # CUDA backend implementations
│   │   ├── Gemm/              # GEMM operation variants
│   │   ├── CublasHelper.hpp
│   │   └── CudaBlasWorkload.hpp
│   ├── hip/                    # HIP backend implementations
│   │   ├── Gemm/              # GEMM operation variants
│   │   ├── RocBlasHelper.hpp
│   │   └── RocBlasWorkload.hpp
│   ├── Buffers.hpp             # Buffer management
│   ├── Random.hpp              # Random data generation
│   ├── Types.hpp               # Type definitions
│   └── Validation.hpp          # Result validation
├── config/                     # Protocol configuration files
├── lines-of-code.py            # LOC analysis script
└── README.md                   # This file
```

## Key Statistics

- **Total source LOC**: ~2,810
- **Cross-backend shared code**: 26% of total
- **Specializations**: 9 GEMM specializations
- **Workloads**: 50 registered workloads per backend
- **Average LOC per specialization**: ~109 lines
- **Average LOC per workload registration**: ~3 lines