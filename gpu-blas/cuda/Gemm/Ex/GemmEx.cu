#include "cublas_v2.h"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <cuda_runtime.h>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/Gemm/Ex/CudaBlasGemmEx.hpp>

namespace GpuBlas {

  namespace {

    using hgemm_f32_acc = CublasGemmEx<Shapes::TypeConfig<__half, float, __half>, int>;
    using hgemm_f32_acc_64 = CublasGemmEx<Shapes::TypeConfig<__half, float, __half>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_acc);
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_acc_64);
    using hgemm_f32_out_f32_acc = CublasGemmEx<Shapes::TypeConfig<__half, float, float>, int>;
    using hgemm_f32_out_f32_acc_64 = CublasGemmEx<Shapes::TypeConfig<__half, float, float>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_out_f32_acc);
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_out_f32_acc_64);

    using bgemm_f32_acc = CublasGemmEx<Shapes::TypeConfig<__nv_bfloat16, float, __nv_bfloat16>, int>;
    using bgemm_f32_acc_64 = CublasGemmEx<Shapes::TypeConfig<__nv_bfloat16, float, __nv_bfloat16>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(bgemm_f32_acc);
    BASELINER_REGISTER_WORKLOAD(bgemm_f32_acc_64);

    using igemm_i32_acc = CublasGemmEx<Shapes::TypeConfig<int8_t, int32_t, int32_t>, int>;
    using igemm_i32_acc_64 = CublasGemmEx<Shapes::TypeConfig<int8_t, int32_t, int32_t>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(igemm_i32_acc);
    BASELINER_REGISTER_WORKLOAD(igemm_i32_acc_64);
    using igemm_i32_out_i32_acc = CublasGemmEx<Shapes::TypeConfig<int8_t, int32_t, int32_t>, int>;
    using igemm_i32_out_i32_acc_64 = CublasGemmEx<Shapes::TypeConfig<int8_t, int32_t, int32_t>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(igemm_i32_out_i32_acc);
    BASELINER_REGISTER_WORKLOAD(igemm_i32_out_i32_acc_64);
  } // namespace
} // namespace GpuBlas
