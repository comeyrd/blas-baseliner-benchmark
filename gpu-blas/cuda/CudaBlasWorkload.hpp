#ifndef BLAS_BASELINER_CUDABLASWORKLOAD_HPP
#define BLAS_BASELINER_CUDABLASWORKLOAD_HPP
#include "../BlasShapes.hpp"
#include "../IBlasWorkload.hpp"
#include "CublasHelper.hpp"
#include "CudaMemoryBackend.hpp"
#include "cublas_v2.h"
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

namespace GpuBlas {

  template <typename ShapeT>
  class CuBlasWorkload : public IBlasWorkload<Baseliner::Hardware::CudaBackend, ShapeT, CudaMemoryBackend> {
  public:
    using backend = typename IBlasWorkload<Baseliner::Hardware::CudaBackend, ShapeT, CudaMemoryBackend>::backend;
    void alloc_handle() override {
      CHECK_CUBLAS(cublasCreate(&m_handle));
    };
    void free_handle() override {
      CHECK_CUBLAS(cublasDestroy(m_handle));
    };
    void run_workload(std::shared_ptr<typename backend::stream_t> stream) override;

  private:
    cublasHandle_t m_handle;
  };

  template <typename TypeConfigT>
  using CublasGemmEx = CuBlasWorkload<Shapes::GemmShape<TypeConfigT>>;

  template <typename TypeT>
  using CublasGemm = CublasGemmEx<Shapes::TypeConfig<TypeT>>;

} // namespace GpuBlas
#endif // BLAS_BASELINER_CUDABLASWORKLOAD_HPP