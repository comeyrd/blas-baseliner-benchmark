#ifndef GPU_BLAS_CUBLAS_HELPER_HPP
#define GPU_BLAS_CUBLAS_HELPER_HPP
#include <rocblas/rocblas.h>

void CHECK_ROCBLAS_error(rocblas_status error_code, const char *file, int line);                // NOLINT
void CHECK_ROCBLAS_error_no_except(rocblas_status error_code, const char *file, int line);      // NOLINT
#define CHECK_ROCBLAS(error) CHECK_ROCBLAS_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_ROCBLAS_NO_EXCEPT(error) CHECK_ROCBLAS_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace GpuBlas {}
#endif // GPU_BLAS_CUBLAS_HELPER_HPP
#
