#ifndef BLAS_BASELINER_HIPMEMORYBACKEND_HPP
#define BLAS_BASELINER_HIPMEMORYBACKEND_HPP

#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <gpu-blas/IMemoryBackend.hpp>
namespace GpuBlas {
  struct HipMemoryBackend : public IMemoryBackend<Baseliner::Hardware::HipBackend> {
    void free(void *ptr) override {
      CHECK_HIP(hipFree(ptr));
    };
    void memcpy_to_device(void *dst, const void *src, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_HIP(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, *stream));
    };
    void memcpy_to_host(void *dst, const void *src, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_HIP(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, *stream));
    };
    void memset(void *ptr, int value, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_HIP(hipMemsetAsync(ptr, value, bytes, *stream));
    };
    HipMemoryBackend() = default;
    ~HipMemoryBackend() = default;

  protected:
    void _malloc(void **ptr, size_t bytes, std::shared_ptr<stream_t> &stream) override {
      CHECK_HIP(hipMallocAsync(ptr, bytes, *stream));
    };
  };
} // namespace GpuBlas

#endif // BLAS_BASELINER_HIPMEMORYBACKEND_HPP