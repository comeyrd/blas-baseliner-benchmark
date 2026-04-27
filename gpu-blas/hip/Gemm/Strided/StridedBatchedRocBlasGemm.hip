#include "rocblas/rocblas.h"
#include <baseliner/registry/RegisteringMacros.hpp>
#include <baseliner/specs/Conversions.hpp>

#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/Gemm/Strided/StridedBatchedRocBlasGemm.hpp>
#include <gpu-blas/hip/RocBlasWorkload.hpp>

namespace GpuBlas {
  namespace {
    using stridedbatchedsgemm = StridedBatchedRocBlasGemm<Types::TypeConfig<float>, int>;
    using stridedbatchedsgemm_64 = StridedBatchedRocBlasGemm<Types::TypeConfig<float>, int64_t>;
    using stridedbatcheddgemm = StridedBatchedRocBlasGemm<Types::TypeConfig<double>, int>;
    using stridedbatcheddgemm_64 = StridedBatchedRocBlasGemm<Types::TypeConfig<double>, int64_t>;
    using stridedbatchedcgemm = StridedBatchedRocBlasGemm<Types::TypeConfig<rocblas_float_complex>, int>;
    using stridedbatchedcgemm_64 = StridedBatchedRocBlasGemm<Types::TypeConfig<rocblas_float_complex>, int64_t>;
    using stridedbatchedzgemm = StridedBatchedRocBlasGemm<Types::TypeConfig<rocblas_double_complex>, int>;
    using stridedbatchedzgemm_64 = StridedBatchedRocBlasGemm<Types::TypeConfig<rocblas_double_complex>, int64_t>;
    using stridedbatchedhgemm = StridedBatchedRocBlasGemm<Types::TypeConfig<rocblas_half>, int>;
    using stridedbatchedhgemm_64 = StridedBatchedRocBlasGemm<Types::TypeConfig<rocblas_half>, int64_t>;

    BASELINER_REGISTER_WORKLOAD(stridedbatchedsgemm);
    BASELINER_REGISTER_WORKLOAD(stridedbatchedsgemm_64);
    BASELINER_REGISTER_WORKLOAD(stridedbatcheddgemm);
    BASELINER_REGISTER_WORKLOAD(stridedbatcheddgemm_64);
    BASELINER_REGISTER_WORKLOAD(stridedbatchedcgemm);
    BASELINER_REGISTER_WORKLOAD(stridedbatchedcgemm_64);
    BASELINER_REGISTER_WORKLOAD(stridedbatchedzgemm);
    BASELINER_REGISTER_WORKLOAD(stridedbatchedzgemm_64);
    BASELINER_REGISTER_WORKLOAD(stridedbatchedhgemm);
    BASELINER_REGISTER_WORKLOAD(stridedbatchedhgemm_64);
  } // namespace

} // namespace GpuBlas