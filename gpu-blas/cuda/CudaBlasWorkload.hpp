#ifndef BLAS_BASELINER_CUDABLASWORKLOAD_HPP
#define BLAS_BASELINER_CUDABLASWORKLOAD_HPP
#include "../BlasShapes.hpp"
#include "../IBlasWorkload.hpp"
#include "CublasHelper.hpp"
#include "CudaMemoryBackend.hpp"
#include "cublas_v2.h"
#include "types.hpp"
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

  protected:
    cublasHandle_t m_handle;
  };

} // namespace GpuBlas
#endif // BLAS_BASELINER_CUDABLASWORKLOAD_HPP