#include "rocblas/rocblas.h"
#include <baseliner/Register.hpp>
#include <baseliner/core/Conversions.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/Gemm/Batched/BatchedRocBlasGemm.hpp>

namespace GpuBlas {
  namespace {
    using batchedsgemm = BatchedRocBlasGemm<Types::TypeConfig<float>, int>;
    using batchedsgemm_64 = BatchedRocBlasGemm<Types::TypeConfig<float>, int64_t>;
    using batcheddgemm = BatchedRocBlasGemm<Types::TypeConfig<double>, int>;
    using batcheddgemm_64 = BatchedRocBlasGemm<Types::TypeConfig<double>, int64_t>;
    using batchedcgemm = BatchedRocBlasGemm<Types::TypeConfig<rocblas_float_complex>, int>;
    using batchedcgemm_64 = BatchedRocBlasGemm<Types::TypeConfig<rocblas_float_complex>, int64_t>;
    using batchedzgemm = BatchedRocBlasGemm<Types::TypeConfig<rocblas_double_complex>, int>;
    using batchedzgemm_64 = BatchedRocBlasGemm<Types::TypeConfig<rocblas_double_complex>, int64_t>;
    using batchedhgemm = BatchedRocBlasGemm<Types::TypeConfig<rocblas_half>, int>;
    using batchedhgemm_64 = BatchedRocBlasGemm<Types::TypeConfig<rocblas_half>, int64_t>;

    BASELINER_REGISTER_WORKLOAD(batchedsgemm);
    BASELINER_REGISTER_WORKLOAD(batchedsgemm_64);
    BASELINER_REGISTER_WORKLOAD(batcheddgemm);
    BASELINER_REGISTER_WORKLOAD(batcheddgemm_64);
    BASELINER_REGISTER_WORKLOAD(batchedcgemm);
    BASELINER_REGISTER_WORKLOAD(batchedcgemm_64);
    BASELINER_REGISTER_WORKLOAD(batchedzgemm);
    BASELINER_REGISTER_WORKLOAD(batchedzgemm_64);
    BASELINER_REGISTER_WORKLOAD(batchedhgemm);
    BASELINER_REGISTER_WORKLOAD(batchedhgemm_64);
  } // namespace

} // namespace GpuBlas