#include "cublas_v2.h"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/Gemm/Batched/BatchedCublasGemm.hpp>

namespace GpuBlas {
  namespace {
    using batchedsgemm = BatchedCublasGemm<Types::TypeConfig<float>, int>;
    using batchedsgemm_64 = BatchedCublasGemm<Types::TypeConfig<float>, int64_t>;
    using batcheddgemm = BatchedCublasGemm<Types::TypeConfig<double>, int>;
    using batcheddgemm_64 = BatchedCublasGemm<Types::TypeConfig<double>, int64_t>;
    using batchedcgemm = BatchedCublasGemm<Types::TypeConfig<cuComplex>, int>;
    using batchedcgemm_64 = BatchedCublasGemm<Types::TypeConfig<cuComplex>, int64_t>;
    using batchedzgemm = BatchedCublasGemm<Types::TypeConfig<cuDoubleComplex>, int>;
    using batchedzgemm_64 = BatchedCublasGemm<Types::TypeConfig<cuDoubleComplex>, int64_t>;
    using batchedhgemm = BatchedCublasGemm<Types::TypeConfig<__half>, int>;
    using batchedhgemm_64 = BatchedCublasGemm<Types::TypeConfig<__half>, int64_t>;

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