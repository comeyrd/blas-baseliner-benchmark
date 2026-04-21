

#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/Gemm/Regular/RegularRocBlasGemm.hpp>
#include <gpu-blas/hip/RocBlasWorkload.hpp>

namespace GpuBlas {
  namespace {
    using sgemm = RegularRocBlasGemm<Types::TypeConfig<rocblas_float>, rocblas_int>;
    using sgemm_64 = RegularRocBlasGemm<Types::TypeConfig<rocblas_float>, int64_t>;
    using dgemm = RegularRocBlasGemm<Types::TypeConfig<rocblas_double>, rocblas_int>;
    using dgemm_64 = RegularRocBlasGemm<Types::TypeConfig<rocblas_double>, int64_t>;
    using cgemm = RegularRocBlasGemm<Types::TypeConfig<rocblas_float_complex>, rocblas_int>;
    using cgemm_64 = RegularRocBlasGemm<Types::TypeConfig<rocblas_float_complex>, int64_t>;
    using zgemm = RegularRocBlasGemm<Types::TypeConfig<rocblas_double_complex>, rocblas_int>;
    using zgemm_64 = RegularRocBlasGemm<Types::TypeConfig<rocblas_double_complex>, int64_t>;
    using hgemm = RegularRocBlasGemm<Types::TypeConfig<rocblas_half>, int>;
    using hgemm_64 = RegularRocBlasGemm<Types::TypeConfig<rocblas_half>, int64_t>;

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