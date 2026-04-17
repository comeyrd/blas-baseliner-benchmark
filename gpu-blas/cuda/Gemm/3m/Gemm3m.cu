#include <baseliner/registry/RegisteringMacros.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/Gemm/3m/CudaBlasGemm3m.hpp>
namespace GpuBlas {

  namespace {

    using Cgemm3m = CublasGemm3m<Shapes::TypeConfig<cuComplex>, int>;
    using Cgemm3m_64 = CublasGemm3m<Shapes::TypeConfig<cuComplex>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(Cgemm3m)
    BASELINER_REGISTER_WORKLOAD(Cgemm3m_64)

    using Zgemm3m = CublasGemm3m<Shapes::TypeConfig<cuDoubleComplex>, int>;
    using Zgemm3m_64 = CublasGemm3m<Shapes::TypeConfig<cuDoubleComplex>, int64_t>;
    BASELINER_REGISTER_WORKLOAD(Zgemm3m)
    BASELINER_REGISTER_WORKLOAD(Zgemm3m_64)

  } // namespace
} // namespace GpuBlas
