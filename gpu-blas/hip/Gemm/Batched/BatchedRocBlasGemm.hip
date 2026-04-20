#include "rocblas/rocblas.h"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/Gemm/Batched/BatchedRocBlasGemm.hpp>

namespace GpuBlas {
  namespace {
    using batchedsgemm = BatchedRocBlasGemm<Shapes::TypeConfig<float>, int>;
    using batchedsgemm_64 = BatchedRocBlasGemm<Shapes::TypeConfig<float>, int64_t>;
    using batcheddgemm = BatchedRocBlasGemm<Shapes::TypeConfig<double>, int>;
    using batcheddgemm_64 = BatchedRocBlasGemm<Shapes::TypeConfig<double>, int64_t>;
    using batchedcgemm = BatchedRocBlasGemm<Shapes::TypeConfig<rocblas_float_complex>, int>;
    using batchedcgemm_64 = BatchedRocBlasGemm<Shapes::TypeConfig<rocblas_float_complex>, int64_t>;
    using batchedzgemm = BatchedRocBlasGemm<Shapes::TypeConfig<rocblas_double_complex>, int>;
    using batchedzgemm_64 = BatchedRocBlasGemm<Shapes::TypeConfig<rocblas_double_complex>, int64_t>;
    using batchedhgemm = BatchedRocBlasGemm<Shapes::TypeConfig<rocblas_half>, int>;
    using batchedhgemm_64 = BatchedRocBlasGemm<Shapes::TypeConfig<rocblas_half>, int64_t>;

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