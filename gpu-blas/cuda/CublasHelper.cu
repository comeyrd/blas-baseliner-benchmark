#include "CublasHelper.hpp"
#include "baseliner/core/Error.hpp"
#include "cublas_v2.h"
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>

void check_cublas_error(cublasStatus_t error_code, const char *file, int line) {
  if (error_code != cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
    throw Baseliner::Errors::hardware_error("CUBLAS", cublasGetStatusString(error_code), file, line);
  }
}; // NOLINT
void check_cublas_error_no_except(cublasStatus_t error_code, const char *file, int line) {
  if (error_code != cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
    std::string msg = std::string("CUBLAS Error : ") + cublasGetStatusString(error_code) + std::string(" in : ") +
                      file + std::string(" line ") + std::to_string(line);
    std::cerr << msg << std::endl;
  }
}; // NOLINT
