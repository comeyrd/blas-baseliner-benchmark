#ifndef BLAS_BASELINER_IMEMORY_BACKEND_HPP
#define BLAS_BASELINER_IMEMORY_BACKEND_HPP
#include <memory>
namespace GpuBlas {
  template <typename HardwareT>
  struct IMemoryBackend {
    using backend_t = HardwareT;
    using stream_t = typename HardwareT::stream_t;
    virtual void malloc(void *&ptr, size_t bytes, std::shared_ptr<stream_t> &stream) = 0;
    virtual void free(void *&ptr) = 0;
    virtual void memcpy_to_device(void *dst, const void *src, size_t bytes, std::shared_ptr<stream_t> &stream) = 0;
    virtual void memcpy_to_host(void *dst, const void *src, size_t bytes, std::shared_ptr<stream_t> &stream) = 0;
    virtual void memset(void *ptr, int value, size_t bytes, std::shared_ptr<stream_t> &stream) = 0;
    IMemoryBackend<HardwareT>() = default;
    virtual ~IMemoryBackend<HardwareT>() = default;
  };
} // namespace GpuBlas
#endif // BLAS_BASELINER_IMEMORY_BACKEND_HPP