#include "CudaBlas.hpp"
#include "cublas_v2.h"
#include "types.hpp"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
namespace GpuBlas {
  template <>
  void CudaGemm<float>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasSgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  template <>
  void CudaGemm<double>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasDgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  template <>
  void CudaGemm<cuComplex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasCgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  template <>
  void CudaGemm<cuDoubleComplex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasZgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  template <>
  void CudaGemm<__half>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasHgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  namespace {
    using sgemm = CudaGemm<float>;
    using dgemm = CudaGemm<double>;
    using cgemm = CudaGemm<cuComplex>;
    using zgemm = CudaGemm<cuDoubleComplex>;
    using hgemm = CudaGemm<__half>;
    BASELINER_REGISTER_WORKLOAD(sgemm);
    BASELINER_REGISTER_WORKLOAD(dgemm);
    BASELINER_REGISTER_WORKLOAD(cgemm);
    BASELINER_REGISTER_WORKLOAD(zgemm);
    BASELINER_REGISTER_WORKLOAD(hgemm);
  } // namespace

} // namespace GpuBlas