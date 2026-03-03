#ifndef GPU_BLAS_CUDA_BLAS_HPP
#define GPU_BLAS_CUDA_BLAS_HPP
#include "../GpuBlas.hpp"
#include "CublasHelper.hpp"
#include "cublas_v2.h"
#include <baseliner/hardware/cuda/CudaBackend.hpp>
namespace GpuBlas {

  // TODO Support Half Precision
  template <typename TypeT>
  class CudaGemm : public IGemm<Baseliner::Hardware::CudaBackend, TypeT> {
  public:
    using backend = typename IGemm<Baseliner::Hardware::CudaBackend, TypeT>::backend;

    void setup(std::shared_ptr<typename backend::stream_t> stream) override {
      this->alloc_host();
      CHECK_CUBLAS(cublasCreate(&handle));
      CHECK_CUDA(cudaMallocAsync(&this->m_d_A, this->m_k * this->m_m * sizeof(TypeT), *stream));
      CHECK_CUDA(cudaMallocAsync(&this->m_d_B, this->m_k * this->m_n * sizeof(TypeT), *stream));
      CHECK_CUDA(cudaMallocAsync(&this->m_d_C, this->m_m * this->m_n * sizeof(TypeT), *stream));

      CHECK_CUDA(cudaMemcpyAsync(this->m_d_A, this->m_A.data(), this->m_k * this->m_m * sizeof(TypeT),
                                 cudaMemcpyHostToDevice, *stream));
      CHECK_CUDA(cudaMemcpyAsync(this->m_d_B, this->m_B.data(), this->m_k * this->m_n * sizeof(TypeT),
                                 cudaMemcpyHostToDevice, *stream));
    };
    void reset_case(std::shared_ptr<typename backend::stream_t> stream) override {
      CHECK_CUDA(cudaMemsetAsync(this->m_d_C, 0, this->m_m * this->m_n * sizeof(TypeT)));
    };
    void run_case(std::shared_ptr<typename backend::stream_t> stream) override;
    void teardown(std::shared_ptr<typename backend::stream_t> stream) override {
      CHECK_CUDA(cudaMemcpyAsync(this->m_C.data(), this->m_d_C, this->m_m * this->m_n * sizeof(TypeT),
                                 cudaMemcpyDeviceToHost, *stream));
    };

  private:
    cublasHandle_t handle;
  };

}; // namespace GpuBlas
#endif // GPU_BLAS_CUDA_BLAS_HPP