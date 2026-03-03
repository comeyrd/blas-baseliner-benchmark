#ifndef GPU_BLAS_CUBLAS_HELPER_HPP
#define GPU_BLAS_CUBLAS_HELPER_HPP
#include "cublas_v2.h"

void check_cublas_error(cublasStatus_t error_code, const char *file, int line);               // NOLINT
void check_cublas_error_no_except(cublasStatus_t error_code, const char *file, int line);     // NOLINT
#define CHECK_CUBLAS(error) check_cublas_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_CUBLAS_NO_EXCEPT(error) check_cublas_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace GpuBlas {}
#endif // GPU_BLAS_CUBLAS_HELPER_HPP
#