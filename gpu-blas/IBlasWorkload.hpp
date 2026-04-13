#ifndef BLAS_BASELINER_IBLASWORKLOAD_HPP
#define BLAS_BASELINER_IBLASWORKLOAD_HPP
#include "Random.hpp"
#include "Types.hpp"
#include <baseliner/core/Workload.hpp>
namespace GpuBlas {

  template <typename ShapeT>
  struct Buffers {
    static constexpr size_t N = ShapeT::buffer_count;

    std::array<void *, N> device{};
    std::array<std::vector<std::byte>, N> host;
    template <size_t I>
    auto *device_ptr() {
      using T = typename ShapeT::template BufferT<I>;
      return static_cast<T *>(device[I]);
    }

    template <size_t I>
    auto &host_vec() {
      using T = typename ShapeT::template BufferT<I>;
      return reinterpret_cast<std::vector<T> &>(host[I]);
    }
  };

  template <typename BackendT, typename ShapeT, typename MemoryBackendT>
  class IBlasWorkload : public Baseliner::IWorkload<BackendT> {
  public:
    using InputT = typename ShapeT::InputT;
    using OutputT = typename ShapeT::OutputT;
    using ComputeT = typename ShapeT::ComputeT;
    using DimsT = typename ShapeT::DimsT;
    using ArgsT = typename ShapeT::ArgsT;

    static constexpr size_t N = ShapeT::buffer_count;

    auto name() -> std::string override {
      return std::string(ShapeT::group) + Types::type_to_string<ShapeT::TypeConfigT>();
    }
    auto validate_workload() -> bool override {
      return true;
    }

    virtual void alloc_handle() = 0;
    virtual void free_handle() = 0;
    void alloc_host() {
      m_dims = ShapeT::scale(this->get_work_size());
      auto sizes = ShapeT::buffer_sizes(m_dims);
      for (size_t i = 0; i < N; ++i) {
        m_buffers.host[i].resize(sizes[i]);
        Random::apply_fill(m_buffers.host[i], ShapeT::fill_policies[i], this->get_seed());
      }
    }

    void setup(std::shared_ptr<typename BackendT::stream_t> stream) override {
      this->alloc_host();
      this->alloc_handle();
      auto sizes = ShapeT::buffer_sizes(m_dims);

      for (size_t i = 0; i < N; ++i) {
        m_memory.malloc(m_buffers.device[i], sizes[i] * sizeof(InputT), stream);
        m_memory.memcpy_to_device(m_buffers.device[i], m_buffers.host[i].data(), sizes[i] * sizeof(InputT), stream);
      }
    }
    void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) override {
      auto sizes = ShapeT::buffer_sizes(m_dims);

      for (size_t i = 0; i < N; ++i) {
        if (ShapeT::is_output[i]) {
          m_memory.memset(m_buffers.device[i], 0, sizes[i] * sizeof(InputT), stream);
        }
      }
    }
    void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override {
      auto sizes = ShapeT::buffer_sizes(m_dims);

      for (size_t i = 0; i < N; ++i) {
        if (ShapeT::is_output[i]) {
          m_memory.memcpy_to_host(m_buffers.host[i].data(), m_buffers.device[i], sizes[i] * sizeof(InputT), stream);
        }
        m_memory.free(m_buffers.device[i]);
      }
      this->free_handle();
      this->free_host();
    }

    void free_host() {
      for (size_t i = 0; i < N; ++i)
        m_buffers.host[i].clear();
    }

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return ShapeT::flop_count(m_dims) * Types::TypeOperations<InputT>::factor;
    }

    auto number_of_bytes() -> std::optional<size_t> override {
      return ShapeT::byte_count(m_dims) * sizeof(InputT);
    }

    void register_options() override {
      Baseliner::IWorkload<BackendT>::register_options();
      ShapeT::register_options(*this, m_dims);
      m_args.register_options(*this);
    }

  protected:
    MemoryBackendT m_memory{};
    DimsT m_dims{};
    ArgsT m_args{};
    Buffers<ShapeT> m_buffers{};
  };
} // namespace GpuBlas
#endif // BLAS_BASELINER_IBLASWORKLOAD_HPP