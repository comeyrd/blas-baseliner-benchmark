#ifndef BLAS_BASELINER_CUDAMEMORYBACKEND_HPP
#define BLAS_BASELINER_CUDAMEMORYBACKEND_HPP
#include "../IMemoryBackend.hpp"
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
namespace GpuBlas {
  struct CudaMemoryBackend : public IMemoryBackend<Baseliner::Hardware::CudaBackend> {
    void malloc(void *&ptr, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_CUDA(cudaMallocAsync(&ptr, bytes, *stream));
    };
    void free(void *&ptr) override {
      CHECK_CUDA(cudaFree(ptr));
    };
    void memcpy_to_device(void *dst, const void *src, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_CUDA(cudaMemcpyAsync(&dst, &src, bytes, cudaMemcpyHostToDevice, *stream));
    };
    void memcpy_to_host(void *dst, const void *src, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_CUDA(cudaMemcpyAsync(&dst, &src, bytes, cudaMemcpyDeviceToHost, *stream));
    };
    void memset(void *ptr, int value, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_CUDA(cudaMemsetAsync(ptr, value, bytes, *stream));
    };
    CudaMemoryBackend() = default;
    ~CudaMemoryBackend() = default;
  };
} // namespace GpuBlas

#endif // BLAS_BASELINER_CUDAMEMORYBACKEND_HPP