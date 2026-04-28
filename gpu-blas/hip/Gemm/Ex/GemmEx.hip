#include "rocblas/rocblas.h"
#include <baseliner/Register.hpp>
#include <baseliner/core/Conversions.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/Gemm/Ex/RocBlasGemmEx.hpp>
namespace GpuBlas {
  namespace {

    // Source :
    // https://rocm.docs.amd.com/projects/rocBLAS/en/develop/reference/extension.html#rocblas-gemm-ex-batched-strided-batched
    //
    //  rocblas_datatype_f64_r = a_type = b_type = c_type = d_type = compute_type
    using dgemm_ex = RocBlasGemmEx<Types::TypeConfig<double, double, double>, int>;
    using dgemm_ex_64 = RocBlasGemmEx<Types::TypeConfig<double, double, double>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(dgemm_ex);
    BASELINER_REGISTER_WORKLOAD(dgemm_ex_64);

    // rocblas_datatype_f32_r = a_type = b_type = c_type = d_type = compute_type
    using sgemm_ex = RocBlasGemmEx<Types::TypeConfig<float, float, float>, int>;
    using sgemm_ex_64 = RocBlasGemmEx<Types::TypeConfig<float, float, float>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(sgemm_ex);
    BASELINER_REGISTER_WORKLOAD(sgemm_ex_64);

    // rocblas_datatype_f16_r = a_type = b_type = c_type = d_type = compute_type
    using hgemm_ex = RocBlasGemmEx<Types::TypeConfig<rocblas_half, rocblas_half, rocblas_half>, int>;
    using hgemm_ex_64 = RocBlasGemmEx<Types::TypeConfig<rocblas_half, rocblas_half, rocblas_half>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(hgemm_ex);
    BASELINER_REGISTER_WORKLOAD(hgemm_ex_64);

    // rocblas_datatype_f16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r = compute_type
    using hgemm_f32_acc = RocBlasGemmEx<Types::TypeConfig<rocblas_half, float, rocblas_half>, int>;
    using hgemm_f32_acc_64 = RocBlasGemmEx<Types::TypeConfig<rocblas_half, float, rocblas_half>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_acc);
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_acc_64);

    // rocblas_datatype_f16_r = a_type = b_type; rocblas_datatype_f32_r = c_type = d_type = compute_type
    using hgemm_f32_out_f32_acc = RocBlasGemmEx<Types::TypeConfig<rocblas_half, float, float>, int>;
    using hgemm_f32_out_f32_acc_64 = RocBlasGemmEx<Types::TypeConfig<rocblas_half, float, float>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_out_f32_acc);
    BASELINER_REGISTER_WORKLOAD(hgemm_f32_out_f32_acc_64);

    // rocblas_datatype_bf16_r = a_type = b_type = c_type = d_type; rocblas_datatype_f32_r = compute_type
    using bf16gemm_f32_acc = RocBlasGemmEx<Types::TypeConfig<rocblas_bfloat16, float, rocblas_bfloat16>, int>;
    using bf16gemm_f32_acc_64 = RocBlasGemmEx<Types::TypeConfig<rocblas_bfloat16, float, rocblas_bfloat16>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(bf16gemm_f32_acc);
    BASELINER_REGISTER_WORKLOAD(bf16gemm_f32_acc_64);

    // rocblas_datatype_bf16_r = a_type = b_type; rocblas_datatype_f32_r = c_type = d_type = compute_type
    using bf16gemm_f32_out_f32_acc = RocBlasGemmEx<Types::TypeConfig<rocblas_bfloat16, float, float>, int>;
    using bf16gemm_f32_out_f32_acc_64 = RocBlasGemmEx<Types::TypeConfig<rocblas_bfloat16, float, float>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(bf16gemm_f32_out_f32_acc);
    BASELINER_REGISTER_WORKLOAD(bf16gemm_f32_out_f32_acc_64);

    // rocblas_datatype_i8_r = a_type = b_type; rocblas_datatype_i32_r = c_type = d_type = compute_type
    using igemm_i32_acc = RocBlasGemmEx<Types::TypeConfig<int8_t, int32_t, int32_t>, int>;
    using igemm_i32_acc_64 = RocBlasGemmEx<Types::TypeConfig<int8_t, int32_t, int32_t>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(igemm_i32_acc);
    BASELINER_REGISTER_WORKLOAD(igemm_i32_acc_64);

    // rocblas_datatype_f32_c = a_type = b_type = c_type = d_type = compute_type
    using cgemm_ex =
        RocBlasGemmEx<Types::TypeConfig<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>, int>;
    using cgemm_ex_64 =
        RocBlasGemmEx<Types::TypeConfig<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(cgemm_ex);
    BASELINER_REGISTER_WORKLOAD(cgemm_ex_64);

    // rocblas_datatype_f64_c = a_type = b_type = c_type = d_type = compute_type
    using zgemm_ex =
        RocBlasGemmEx<Types::TypeConfig<rocblas_double_complex, rocblas_double_complex, rocblas_double_complex>, int>;
    using zgemm_ex_64 =
        RocBlasGemmEx<Types::TypeConfig<rocblas_double_complex, rocblas_double_complex, rocblas_double_complex>,
                      int64_t>;
    BASELINER_REGISTER_WORKLOAD(zgemm_ex);
    BASELINER_REGISTER_WORKLOAD(zgemm_ex_64);

  } // namespace
} // namespace GpuBlas