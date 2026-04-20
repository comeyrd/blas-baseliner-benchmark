#ifndef BLAS_BASELINER_IBLASWORKLOAD_HPP
#define BLAS_BASELINER_IBLASWORKLOAD_HPP
#include <baseliner/core/Workload.hpp>
#include <gpu-blas/Random.hpp>
#include <gpu-blas/Types.hpp>
namespace GpuBlas {

  template <typename ShapeT>
  struct Buffers {
    static constexpr size_t I_ = ShapeT::input_counts;
    static constexpr size_t O_ = ShapeT::output_counts;

    std::array<typename ShapeT::InputT *, I_> input_device{};
    std::array<std::vector<typename ShapeT::InputT>, I_> input_host;
    std::array<typename ShapeT::OutputT *, O_> output_device{};
    std::array<std::vector<typename ShapeT::OutputT>, O_> output_host;

    auto in_device(size_t i) -> typename ShapeT::InputT * {
      return static_cast<typename ShapeT::InputT *>(input_device[i]);
    }
    auto out_device(size_t i) -> typename ShapeT::OutputT * {
      return static_cast<typename ShapeT::OutputT *>(output_device[i]);
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

    static constexpr size_t I_ = ShapeT::input_counts;
    static constexpr size_t O_ = ShapeT::output_counts;

    auto name() -> std::string override {
      return std::string(ShapeT::group) + Types::type_to_string<typename ShapeT::TypeConfigT>();
    }
    auto validate_workload() -> bool override {
      return m_valid;
    }

    virtual void alloc_handle() = 0;
    virtual void free_handle() = 0;
    virtual void alloc_host() {
      ShapeT::scale(m_dims, this->get_work_size());
      auto input_sz = ShapeT::input_buffer_sizes(m_dims);
      for (size_t i = 0; i < I_; ++i) {
        m_buffers.input_host[i].resize(input_sz[i]);
        Random::apply_fill(m_buffers.input_host[i], ShapeT::input_fill_policies[i], this->get_seed());
      }
      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < O_; i++) {
        m_buffers.output_host[i].resize(output_sz[i]);
        Random::apply_fill(m_buffers.output_host[i], ShapeT::output_fill_policies[i], this->get_seed());
      }
    }

    void setup(std::shared_ptr<typename BackendT::stream_t> stream) override {
      this->alloc_host();
      this->alloc_handle();
      auto input_sz = ShapeT::input_buffer_sizes(m_dims);
      for (size_t i = 0; i < I_; ++i) {
        m_memory.malloc(&m_buffers.input_device[i], input_sz[i] * sizeof(InputT), stream);
        m_memory.memcpy_to_device(m_buffers.input_device[i], m_buffers.input_host[i].data(),
                                  input_sz[i] * sizeof(InputT), stream);
      }
      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < O_; i++) {
        m_memory.malloc(&m_buffers.output_device[i], output_sz[i] * sizeof(OutputT), stream);
        m_memory.memcpy_to_device(m_buffers.output_device[i], m_buffers.output_host[i].data(),
                                  output_sz[i] * sizeof(OutputT), stream);
      }
    }
    void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) override {
      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < O_; i++) {
        m_memory.memset(m_buffers.output_device[i], 0, output_sz[i] * sizeof(OutputT), stream);
      }
    }
    void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override {
      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < I_; ++i) {
        m_memory.free(m_buffers.input_device[i]);
      }
      for (size_t i = 0; i < O_; i++) {
        m_memory.memcpy_to_host(m_buffers.output_host[i].data(), m_buffers.output_device[i],
                                output_sz[i] * sizeof(OutputT), stream);
      }
      BackendT::synchronize(stream);
      for (size_t i = 0; i < O_; i++) {
        m_memory.free(m_buffers.output_device[i]);
      }
      this->free_handle();
      this->inner_validate();
      this->free_host();
    }

    virtual void free_host() {
      for (size_t i = 0; i < I_; ++i) {
        m_buffers.input_host[i].clear();
      }
      for (size_t i = 0; i < O_; ++i) {
        m_buffers.output_host[i].clear();
      }
    }

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return ShapeT::flop_count(m_dims) * Types::TypeOperations<InputT>::factor;
    }

    auto number_of_bytes() -> std::optional<size_t> override {
      return ShapeT::byte_count(m_dims) * sizeof(InputT);
    }

    void register_options() override {
      Baseliner::IWorkload<BackendT>::register_options();
    }
    void register_options_dependencies() override {
      this->register_consumer(&m_dims);
      this->register_consumer(&m_args);
    }
    virtual void inner_validate() {
      m_valid = true;

      for (size_t i = 0; i < O_; i++) {
        auto &out = m_buffers.output_host[i];
        if (out.empty())
          continue;

        const uint8_t *out_ptr = reinterpret_cast<const uint8_t *>(out.data());
        size_t out_bytes = out.size() * sizeof(typename ShapeT::OutputT);
        size_t check_bytes = std::min(out_bytes, size_t(256));

        // Check if output is just a block of zeros
        uint64_t out_hash = 0;
        for (size_t j = 0; j < check_bytes; ++j) {
          out_hash += out_ptr[j];
        }

        if (out_hash == 0) {
          m_valid = false;
          std::cout << "Wall of zeros\n";
          return;
        }

        // Check if output bytes match any input bytes (shallow copy/stale data)
        for (size_t j = 0; j < I_; ++j) {
          auto &in = m_buffers.input_host[j];
          const uint8_t *in_ptr = reinterpret_cast<const uint8_t *>(in.data());
          size_t in_bytes = in.size() * sizeof(typename ShapeT::InputT);

          if (in_bytes < check_bytes)
            continue;

          uint64_t in_hash = 0;
          for (size_t k = 0; k < check_bytes; ++k) {
            in_hash += in_ptr[k];
          }

          if (out_hash == in_hash) {
            std::cout << "here\n";
            m_valid = false;
            return;
          }
        }
      }
    }

  protected:
    MemoryBackendT m_memory{};
    DimsT m_dims{};
    ArgsT m_args{};
    Buffers<ShapeT> m_buffers{};
    bool m_valid{false};
  };
} // namespace GpuBlas
#endif // BLAS_BASELINER_IBLASWORKLOAD_HPP