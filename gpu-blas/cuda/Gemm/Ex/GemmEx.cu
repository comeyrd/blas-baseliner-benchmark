#include "cublas_v2.h"
#include <baseliner/registry/RegisteringMacros.hpp>
#include <baseliner/specs/Conversions.hpp>
#include <cuda_bf16.h> // Required for __nv_bfloat16
#include <cuda_runtime.h>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/Gemm/Ex/CudaBlasGemmEx.hpp>

namespace GpuBlas {

  namespace {

    // Source
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex

    // TODO better coverage
    //  --- 16-bit Floating Point (Compute 16F) ---
    //  Pure half: A/B: half, Out: half, Acc: half
    using hgemm_h16_acc = CublasGemmEx<Types::TypeConfig<__half, __half, __half>, int>;
    using hgemm_h16_acc_64 = CublasGemmEx<Types::TypeConfig<__half, __half, __half>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(hgemm_h16_acc);
    BASELINER_REGISTER_WORKLOAD(hgemm_h16_acc_64);

    using hgemm_h16_acc_pedantic = CublasGemmEx<Types::TypeConfig<__half, __half, __half, PedanticMath>, int>;
    BASELINER_REGISTER_WORKLOAD(hgemm_h16_acc_pedantic);

    // --- BFloat16 (Compute 32F) ---
    using bfgemm_f32_acc = CublasGemmEx<Types::TypeConfig<__nv_bfloat16, float, __nv_bfloat16>, int>;
    using bfgemm_f32_acc_64 = CublasGemmEx<Types::TypeConfig<__nv_bfloat16, float, __nv_bfloat16>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(bfgemm_f32_acc);
    BASELINER_REGISTER_WORKLOAD(bfgemm_f32_acc_64);

    // Mixed BFloat: A/B: bf16, Out: f32, Acc: float
    using bfgemm_f32_out_f32_acc = CublasGemmEx<Types::TypeConfig<__nv_bfloat16, float, float>, int>;
    BASELINER_REGISTER_WORKLOAD(bfgemm_f32_out_f32_acc);

    using igemm_i32_acc = CublasGemmEx<Types::TypeConfig<int8_t, int32_t, int32_t>, int>;
    using igemm_i32_acc_64 = CublasGemmEx<Types::TypeConfig<int8_t, int32_t, int32_t>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(igemm_i32_acc);
    BASELINER_REGISTER_WORKLOAD(igemm_i32_acc_64);

    // --- Mixed Precision (8-bit Input, Float Compute) ---
    using igemm_f32_out_f32_acc = CublasGemmEx<Types::TypeConfig<int8_t, float, float>, int>;
    BASELINER_REGISTER_WORKLOAD(igemm_f32_out_f32_acc);

    using sgemm_f32_acc = CublasGemmEx<Types::TypeConfig<float, float, float>, int>;
    BASELINER_REGISTER_WORKLOAD(sgemm_f32_acc);

    // TF32 Fast path (Common for Ampere+ GPUs)
    using sgemm_f32_fast_tf32 = CublasGemmEx<Types::TypeConfig<float, float, float, FastTF32Math>, int>;
    BASELINER_REGISTER_WORKLOAD(sgemm_f32_fast_tf32);

    // Fast 16F path (Down-converts internally)
    using sgemm_f32_fast_16f = CublasGemmEx<Types::TypeConfig<float, float, float, Fast16FMath>, int>;
    BASELINER_REGISTER_WORKLOAD(sgemm_f32_fast_16f);

    // --- Double Precision (Compute 64F) ---
    using dgemm_f64_acc = CublasGemmEx<Types::TypeConfig<double, double, double>, int>;
    using dgemm_f64_acc_pedantic = CublasGemmEx<Types::TypeConfig<double, double, double, PedanticMath>, int>;
    BASELINER_REGISTER_WORKLOAD(dgemm_f64_acc);
    BASELINER_REGISTER_WORKLOAD(dgemm_f64_acc_pedantic);

    // --- Complex Single Precision (Compute 32F) ---
    using cgemm_c32_acc = CublasGemmEx<Types::TypeConfig<cuComplex, float, cuComplex>, int>;
    BASELINER_REGISTER_WORKLOAD(cgemm_c32_acc);

    // --- Complex Double Precision (Compute 64F) ---
    using zgemm_c64_acc = CublasGemmEx<Types::TypeConfig<cuDoubleComplex, double, cuDoubleComplex>, int>;
    BASELINER_REGISTER_WORKLOAD(zgemm_c64_acc);
  } // namespace
} // namespace GpuBlas