#include "RocBlasHelper.hpp"
#include <iostream>
#include <ostream>
#include <rocblas/rocblas.h>
#include <stdexcept>
#include <string>

void CHECK_ROCBLAS_error(rocblas_status error_code, const char *file, int line) {
  if (error_code != rocblas_status::rocblas_status_success) {
    std::string msg = std::string("ROCBLAS Error : ") + rocblas_status_to_string(error_code) + std::string(" in : ") +
                      file + std::string(" line ") + std::to_string(line);
    throw std::runtime_error(msg);
  }
}; // NOLINT
void CHECK_ROCBLAS_error_no_except(rocblas_status error_code, const char *file, int line) {
  if (error_code != rocblas_status::rocblas_status_success) {
    std::string msg = std::string("ROCBLAS Error : ") + rocblas_status_to_string(error_code) + std::string(" in : ") +
                      file + std::string(" line ") + std::to_string(line);
    std::cerr << msg << std::endl;
  }
}; // NOLINT
