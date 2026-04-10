#ifndef GPU_BLAS_HIP_BLAS_HPP
#define GPU_BLAS_HIP_BLAS_HPP
#include "../GpuBlas.hpp"
#include "RocBlasHelper.hpp"
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <rocblas/rocblas.h>
namespace GpuBlas {

  // TODO Support Half Precision
  template <typename TypeT>
  class HipGemm : public IGemm<Baseliner::Hardware::HipBackend, TypeT> {
  public:
    using backend = typename IGemm<Baseliner::Hardware::HipBackend, TypeT>::backend;

    void setup(std::shared_ptr<typename backend::stream_t> stream) override {
      this->alloc_host();
      CHECK_ROCBLAS(rocblas_create_handle(&handle));
      CHECK_HIP(hipMallocAsync(&this->m_d_A, this->m_k * this->m_m * sizeof(TypeT), *stream));
      CHECK_HIP(hipMallocAsync(&this->m_d_B, this->m_k * this->m_n * sizeof(TypeT), *stream));
      CHECK_HIP(hipMallocAsync(&this->m_d_C, this->m_m * this->m_n * sizeof(TypeT), *stream));

      CHECK_HIP(hipMemcpyAsync(this->m_d_A, this->m_A.data(), this->m_k * this->m_m * sizeof(TypeT),
                               hipMemcpyHostToDevice, *stream));
      CHECK_HIP(hipMemcpyAsync(this->m_d_B, this->m_B.data(), this->m_k * this->m_n * sizeof(TypeT),
                               hipMemcpyHostToDevice, *stream));
    };
    void reset_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      CHECK_HIP(hipMemsetAsync(this->m_d_C, 0, this->m_m * this->m_n * sizeof(TypeT)));
    };
    void run_workload(std::shared_ptr<typename backend::stream_t> stream) override;
    void teardown(std::shared_ptr<typename backend::stream_t> stream) override {
      CHECK_HIP(hipMemcpyAsync(this->m_C.data(), this->m_d_C, this->m_m * this->m_n * sizeof(TypeT),
                               hipMemcpyDeviceToHost, *stream));
    };

  private:
    rocblas_handle handle;
  };

}; // namespace GpuBlas
#endif // GPU_BLAS_HIP_BLAS_HPP
