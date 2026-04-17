#ifndef BLAS_BASELINER_HIPBLASWORKLOAD_HPP
#define BLAS_BASELINER_HIPBLASWORKLOAD_HPP

#include <gpu-blas/IBatchedBlasWorkload.hpp>
#include <gpu-blas/IBlasWorkload.hpp>
#include <gpu-blas/IStridedBatchedBlasWorkload.hpp>
#include <gpu-blas/hip/HipMemoryBackend.hpp>
#include <gpu-blas/hip/RocBlasHelper.hpp>

#include "rocblas/rocblas.h"
#include <gpu-blas/hip/Types.hpp>

#include <baseliner/core/hardware/hip/HipBackend.hpp>

namespace GpuBlas {

  template <typename ShapeT>
  class CuBlasWorkload : public IBlasWorkload<Baseliner::Hardware::HipBackend, ShapeT, HipMemoryBackend> {
  public:
    using backend = typename IBlasWorkload<Baseliner::Hardware::HipBackend, ShapeT, HipMemoryBackend>::backend;
    void alloc_handle() override {
      CHECK_ROCBLAS(rocblas_create_handle(&m_handle));
    };
    void free_handle() override {
      CHECK_ROCBLAS(rocblas_destroy_handle(m_handle));
    };

  protected:
    rocblas_handle m_handle;
  };

  template <typename ShapeT>
  class BatchedCuBlasWorkload : public IBatchedBlasWorkload<Baseliner::Hardware::HipBackend, ShapeT, HipMemoryBackend> {
  public:
    using backend = typename IBlasWorkload<Baseliner::Hardware::HipBackend, ShapeT, HipMemoryBackend>::backend;
    void alloc_handle() override {
      CHECK_ROCBLAS(rocblas_create_handle(&m_handle));
    };
    void free_handle() override {
      CHECK_ROCBLAS(rocblas_destroy_handle(m_handle));
    };

  protected:
    rocblas_handle m_handle;
  };

  template <typename ShapeT>
  class StridedBatchedCuBlasWorkload
      : public IStridedBatchedBlasWorkload<Baseliner::Hardware::HipBackend, ShapeT, HipMemoryBackend> {
  public:
    using backend = typename IBlasWorkload<Baseliner::Hardware::HipBackend, ShapeT, HipMemoryBackend>::backend;

    void alloc_handle() override {
      CHECK_ROCBLAS(rocblas_create_handle(&m_handle));
    };
    void free_handle() override {
      CHECK_ROCBLAS(rocblas_destroy_handle(m_handle));
    };

  protected:
    rocblas_handle m_handle;
  };

} // namespace GpuBlas
#endif // BLAS_BASELINER_HIPBLASWORKLOAD_HPP