#pragma once
#include <baseliner/core/Workload.hpp>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/Random.hpp>
#include <gpu-blas/Stats.hpp>
#include <gpu-blas/Types.hpp>

namespace GpuBlas {

  struct NonBatched {
    static constexpr size_t default_batch_count = 1;
    static constexpr std::string_view specialization_prefix = "";
  };

  struct StridedBatched {
    static constexpr size_t default_batch_count = 5;
    static constexpr std::string_view specialization_prefix = "strided";
  };

  struct PointerArrayBatched {
    static constexpr size_t default_batch_count = 5;
    static constexpr std::string_view specialization_prefix = "batched";
  };

  template <typename BackendT, typename ShapeT, typename MemoryBackendT, typename BatchPolicy = NonBatched>
  class IBlasWorkload : public Baseliner::IWorkload<BackendT> {
  public:
    using InputT = typename ShapeT::InputT;
    using OutputT = typename ShapeT::OutputT;
    using ComputeT = typename ShapeT::ComputeT;
    using DimsT = typename ShapeT::DimsT;
    using ArgsT = typename ShapeT::ArgsT;

    static constexpr size_t I_ = ShapeT::input_counts;
    static constexpr size_t O_ = ShapeT::output_counts;

    static constexpr bool is_pointer_array = std::is_same_v<BatchPolicy, PointerArrayBatched>;

    std::array<InputT **, I_> input_ptr_arrays{};
    std::array<OutputT **, O_> output_ptr_arrays{};

    auto in_device_array(size_t i) -> InputT ** {
      return input_ptr_arrays[i];
    }
    auto out_device_array(size_t i) -> OutputT ** {
      return output_ptr_arrays[i];
    }

    auto get_stride_in(size_t i) const -> long long int {
      return static_cast<long long int>(ShapeT::input_buffer_sizes(m_dims)[i]);
    }
    auto get_stride_out(size_t i) const -> long long int {
      return static_cast<long long int>(ShapeT::output_buffer_sizes(m_dims)[i]);
    }

    auto algo() -> std::string override {
      return std::string(ShapeT::group);
    }
    auto specialization() -> std::string override {
      return std::string(BatchPolicy::specialization_prefix) +
             Types::type_config_name<typename ShapeT::TypeConfigT, typename ShapeT::DimTypesT>();
    }

    void register_options() override {
      Baseliner::IWorkload<BackendT>::register_options();
      if constexpr (!std::is_same_v<BatchPolicy, NonBatched>)
        this->add_option("Batched", "batch_count", "Number of batches", m_batch_count);
    }
    void register_options_dependencies() override {
      this->register_consumer(&m_base_dims);
      this->register_consumer(&m_args);
    }

    virtual void alloc_handle() = 0;
    virtual void free_handle() = 0;

    // ── Host lifecycle ────────────────────────────────────────────────────
    void setup_host() override {
      m_dims = m_base_dims;
      ShapeT::scale(m_dims, this->get_work_size(), m_batch_count);

      auto input_sz = ShapeT::input_buffer_sizes(m_dims);
      for (size_t i = 0; i < I_; ++i) {
        m_buffers.input_host[i].resize(input_sz[i] * m_batch_count);
        for (size_t b = 0; b < m_batch_count; ++b) {
          auto *ptr = m_buffers.input_host[i].data() + b * input_sz[i];
          Random::apply_fill(ptr, input_sz[i], ShapeT::input_fill_policies[i], this->get_seed() + b);
        }
      }

      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < O_; ++i) {
        m_buffers.output_host[i].resize(output_sz[i] * m_batch_count);
        for (size_t b = 0; b < m_batch_count; ++b) {
          auto *ptr = m_buffers.output_host[i].data() + b * output_sz[i];
          Random::apply_fill(ptr, output_sz[i], ShapeT::output_fill_policies[i], this->get_seed() + b);
        }
      }
    }

    void free() override {
      for (size_t i = 0; i < I_; ++i) {
        m_buffers.input_host[i].clear();
        m_buffers.input_host[i].shrink_to_fit();
      }
      for (size_t i = 0; i < O_; ++i) {
        m_buffers.output_host[i].clear();
        m_buffers.output_host[i].shrink_to_fit();
      }
    }

    void setup_device(typename BackendT::stream_t stream) override {
      this->alloc_handle();
      auto input_sz = ShapeT::input_buffer_sizes(m_dims);
      for (size_t i = 0; i < I_; ++i) {
        m_memory.malloc(&m_buffers.input_device[i], input_sz[i] * m_batch_count * sizeof(InputT), stream);
        m_memory.memcpy_to_device(m_buffers.input_device[i], m_buffers.input_host[i].data(),
                                  input_sz[i] * m_batch_count * sizeof(InputT), stream);

        // NonBatched/Strided: point directly at the flat buffer (no extra allocation).
        // PointerArrayBatched: build a proper device-side pointer array.
        if constexpr (is_pointer_array) {
          std::vector<InputT *> ptrs(m_batch_count);
          for (size_t b = 0; b < m_batch_count; ++b)
            ptrs[b] = m_buffers.input_device[i] + b * input_sz[i];
          m_memory.malloc(&input_ptr_arrays[i], m_batch_count * sizeof(InputT *), stream);
          m_memory.memcpy_to_device(input_ptr_arrays[i], ptrs.data(), m_batch_count * sizeof(InputT *), stream);
        } else {
          input_ptr_arrays[i] = &m_buffers.input_device[i];
        }
      }

      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < O_; ++i) {
        m_memory.malloc(&m_buffers.output_device[i], output_sz[i] * m_batch_count * sizeof(OutputT), stream);
        m_memory.memcpy_to_device(m_buffers.output_device[i], m_buffers.output_host[i].data(),
                                  output_sz[i] * m_batch_count * sizeof(OutputT), stream);
        // NonBatched/Strided: point directly at the flat buffer (no extra allocation).
        // PointerArrayBatched: build a proper device-side pointer array.
        if constexpr (is_pointer_array) {
          std::vector<OutputT *> ptrs(m_batch_count);
          for (size_t b = 0; b < m_batch_count; ++b)
            ptrs[b] = m_buffers.output_device[i] + b * output_sz[i];
          m_memory.malloc(&output_ptr_arrays[i], m_batch_count * sizeof(OutputT *), stream);
          m_memory.memcpy_to_device(output_ptr_arrays[i], ptrs.data(), m_batch_count * sizeof(OutputT *), stream);
        } else {
          output_ptr_arrays[i] = &m_buffers.output_device[i];
        }
      }
    }

    void reset_device(typename BackendT::stream_t stream) override {
      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < O_; ++i)
        m_memory.memset(m_buffers.output_device[i], 0, output_sz[i] * m_batch_count * sizeof(OutputT), stream);
    }

    void fetch_results(typename BackendT::stream_t stream) override {
      if constexpr (is_pointer_array) {
        for (size_t i = 0; i < I_; ++i)
          m_memory.free(input_ptr_arrays[i]);
        for (size_t i = 0; i < O_; ++i)
          m_memory.free(output_ptr_arrays[i]);
      }
      // NonBatched/Strided: ptr arrays point into m_buffers, nothing extra to free.

      auto output_sz = ShapeT::output_buffer_sizes(m_dims);
      for (size_t i = 0; i < I_; ++i)
        m_memory.free(m_buffers.input_device[i]);
      for (size_t i = 0; i < O_; ++i)
        m_memory.memcpy_to_host(m_buffers.output_host[i].data(), m_buffers.output_device[i],
                                output_sz[i] * m_batch_count * sizeof(OutputT), stream);
      BackendT::synchronize(stream);
      for (size_t i = 0; i < O_; ++i)
        m_memory.free(m_buffers.output_device[i]);
      this->free_handle();
    }

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return ShapeT::flop_count(m_dims, m_batch_count);
    }
    auto number_of_bytes() -> std::optional<size_t> override {
      return ShapeT::byte_count(m_dims, m_batch_count);
    }

    void inner_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
      engine->register_metric<MeanError>(m_mean_error);
    }
    void inner_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
      engine->update_values<MeanError>(m_mean_error);
    }

    auto validate() -> bool override {
      return ShapeT::validate(m_buffers, m_dims, m_args, m_mean_error, m_batch_count);
    }

  protected:
    float m_mean_error{};
    MemoryBackendT m_memory{};
    DimsT m_base_dims{};
    DimsT m_dims{};
    ArgsT m_args{};
    Buffers<ShapeT> m_buffers{};
    size_t m_batch_count{BatchPolicy::default_batch_count};
  };

  template <typename BackendT, typename ShapeT, typename MemoryBackendT>
  using IStridedBatchedBlasWorkload = IBlasWorkload<BackendT, ShapeT, MemoryBackendT, StridedBatched>;

  template <typename BackendT, typename ShapeT, typename MemoryBackendT>
  using IBatchedBlasWorkload = IBlasWorkload<BackendT, ShapeT, MemoryBackendT, PointerArrayBatched>;

} // namespace GpuBlas