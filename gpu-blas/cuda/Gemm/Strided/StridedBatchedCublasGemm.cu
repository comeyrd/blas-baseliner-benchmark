#include "cublas_v2.h"
#include <baseliner/registry/RegisteringMacros.hpp>
#include <baseliner/specs/Conversions.hpp>

#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/CudaBlasWorkload.hpp>
#include <gpu-blas/cuda/Gemm/Strided/StridedBatchedCublasGemm.hpp>

namespace GpuBlas {
  namespace {
    using stridedbatchedsgemm = StridedBatchedCublasGemm<Types::TypeConfig<float>, int>;
    using stridedbatchedsgemm_64 = StridedBatchedCublasGemm<Types::TypeConfig<float>, int64_t>;
    using stridedbatcheddgemm = StridedBatchedCublasGemm<Types::TypeConfig<double>, int>;
    using stridedbatcheddgemm_64 = StridedBatchedCublasGemm<Types::TypeConfig<double>, int64_t>;
    using stridedbatchedcgemm = StridedBatchedCublasGemm<Types::TypeConfig<cuComplex>, int>;
    using stridedbatchedcgemm_64 = StridedBatchedCublasGemm<Types::TypeConfig<cuComplex>, int64_t>;
    using stridedbatchedzgemm = StridedBatchedCublasGemm<Types::TypeConfig<cuDoubleComplex>, int>;
    using stridedbatchedzgemm_64 = StridedBatchedCublasGemm<Types::TypeConfig<cuDoubleComplex>, int64_t>;
    using stridedbatchedhgemm = StridedBatchedCublasGemm<Types::TypeConfig<__half>, int>;
    using stridedbatchedhgemm_64 = StridedBatchedCublasGemm<Types::TypeConfig<__half>, int64_t>;

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