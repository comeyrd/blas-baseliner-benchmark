

#include "cublas_v2.h"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/CudaBlasWorkload.hpp>
#include <gpu-blas/cuda/Gemm/Regular/RegularCuBlasGemm.hpp>

namespace GpuBlas {
  namespace {
    using sgemm = RegularCublasGemm<Shapes::TypeConfig<float>, int>;
    using sgemm_64 = RegularCublasGemm<Shapes::TypeConfig<float>, int64_t>;
    using dgemm = RegularCublasGemm<Shapes::TypeConfig<double>, int>;
    using dgemm_64 = RegularCublasGemm<Shapes::TypeConfig<double>, int64_t>;
    using cgemm = RegularCublasGemm<Shapes::TypeConfig<cuComplex>, int>;
    using cgemm_64 = RegularCublasGemm<Shapes::TypeConfig<cuComplex>, int64_t>;
    using zgemm = RegularCublasGemm<Shapes::TypeConfig<cuDoubleComplex>, int>;
    using zgemm_64 = RegularCublasGemm<Shapes::TypeConfig<cuDoubleComplex>, int64_t>;
    using hgemm = RegularCublasGemm<Shapes::TypeConfig<__half>, int>;
    using hgemm_64 = RegularCublasGemm<Shapes::TypeConfig<__half>, int64_t>;

    BASELINER_REGISTER_WORKLOAD(sgemm);
    BASELINER_REGISTER_WORKLOAD(sgemm_64);
    BASELINER_REGISTER_WORKLOAD(dgemm);
    BASELINER_REGISTER_WORKLOAD(dgemm_64);
    BASELINER_REGISTER_WORKLOAD(cgemm);
    BASELINER_REGISTER_WORKLOAD(cgemm_64);
    BASELINER_REGISTER_WORKLOAD(zgemm);
    BASELINER_REGISTER_WORKLOAD(zgemm_64);
    BASELINER_REGISTER_WORKLOAD(hgemm);
    BASELINER_REGISTER_WORKLOAD(hgemm_64);
  } // namespace

} // namespace GpuBlas