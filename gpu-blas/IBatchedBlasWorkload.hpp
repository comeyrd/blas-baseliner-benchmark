#pragma once
#include <gpu-blas/IBlasWorkload.hpp>
namespace GpuBlas {

  template <typename BackendT, typename ShapeT, typename MemoryBackendT>
  class IBatchedBlasWorkload : public IBlasWorkload<BackendT, ShapeT, MemoryBackendT> {
  public:
    using Base = IBlasWorkload<BackendT, ShapeT, MemoryBackendT>;
    using InputT = typename ShapeT::InputT;
    using OutputT = typename ShapeT::OutputT;

    static constexpr size_t I_ = ShapeT::input_counts;
    static constexpr size_t O_ = ShapeT::output_counts;

    std::array<InputT **, I_> input_ptr_arrays{};
    std::array<OutputT **, O_> output_ptr_arrays{};

    auto in_device_array(size_t i) -> InputT ** {
      return input_ptr_arrays[i];
    }
    auto out_device_array(size_t i) -> OutputT ** {
      return output_ptr_arrays[i];
    }

    void register_options() override {
      Base::register_options();
      this->add_option("Batched", "batch_count", "Number of batches", m_batch_count);
    }

    void alloc_host() override {
      this->m_dims = ShapeT::scale(this->get_work_size());
      auto input_sz = ShapeT::input_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < I_; ++i) {
        this->m_buffers.input_host[i].resize(input_sz[i] * m_batch_count);
        // Fill each batch slice independently with a different seed
        size_t slice = input_sz[i];
        for (size_t b = 0; b < m_batch_count; ++b) {
          auto *ptr = this->m_buffers.input_host[i].data() + b * slice;
          Random::apply_fill(ptr, slice, ShapeT::input_fill_policies[i], this->get_seed() + b);
        }
      }
      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < O_; i++) {
        this->m_buffers.output_host[i].resize(output_sz[i] * m_batch_count);
        size_t slice = output_sz[i];
        for (size_t b = 0; b < m_batch_count; ++b) {
          auto *ptr = this->m_buffers.output_host[i].data() + b * slice;
          Random::apply_fill(ptr, slice, ShapeT::input_fill_policies[i], this->get_seed() + b);
        }
      }
    }

    void setup(std::shared_ptr<typename BackendT::stream_t> stream) override {
      this->alloc_host();
      this->alloc_handle();

      auto input_sz = ShapeT::input_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < I_; ++i) {
        // Allocate flat buffer for all batches
        this->m_memory.malloc(&this->m_buffers.input_device[i], input_sz[i] * m_batch_count * sizeof(InputT), stream);
        this->m_memory.memcpy_to_device(this->m_buffers.input_device[i], this->m_buffers.input_host[i].data(),
                                        input_sz[i] * m_batch_count * sizeof(InputT), stream);
        // Build and upload pointer array
        std::vector<InputT *> ptrs(m_batch_count);
        for (size_t b = 0; b < m_batch_count; ++b)
          ptrs[b] = this->m_buffers.in_device(i) + b * input_sz[i];
        this->m_memory.malloc(&input_ptr_arrays[i], m_batch_count * sizeof(InputT *), stream);
        this->m_memory.memcpy_to_device(input_ptr_arrays[i], ptrs.data(), m_batch_count * sizeof(InputT *), stream);
      }

      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < O_; i++) {
        this->m_memory.malloc(&this->m_buffers.output_device[i], output_sz[i] * m_batch_count * sizeof(OutputT),
                              stream);
        this->m_memory.memcpy_to_device(this->m_buffers.output_device[i], this->m_buffers.output_host[i].data(),
                                        output_sz[i] * m_batch_count * sizeof(OutputT), stream);
        std::vector<OutputT *> ptrs(m_batch_count);
        for (size_t b = 0; b < m_batch_count; ++b)
          ptrs[b] = this->m_buffers.out_device(i) + b * output_sz[i];
        this->m_memory.malloc(&output_ptr_arrays[i], m_batch_count * sizeof(OutputT *), stream);
        this->m_memory.memcpy_to_device(output_ptr_arrays[i], ptrs.data(), m_batch_count * sizeof(OutputT *), stream);
      }
    }

    void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) override {
      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < O_; i++)
        this->m_memory.memset(this->m_buffers.output_device[i], 0, output_sz[i] * m_batch_count * sizeof(OutputT),
                              stream);
    }

    void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override {
      // Free pointer arrays first
      for (size_t i = 0; i < I_; ++i)
        this->m_memory.free(input_ptr_arrays[i]);
      for (size_t i = 0; i < O_; ++i)
        this->m_memory.free(output_ptr_arrays[i]);

      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < I_; ++i)
        this->m_memory.free(this->m_buffers.input_device[i]);
      for (size_t i = 0; i < O_; i++) {
        this->m_memory.memcpy_to_host(this->m_buffers.output_host[i].data(), this->m_buffers.output_device[i],
                                      output_sz[i] * m_batch_count * sizeof(OutputT), stream);
        this->m_memory.free(this->m_buffers.output_device[i]);
      }
      this->free_handle();
      this->free_host();
    }

    void free_host() override {
      for (size_t i = 0; i < I_; ++i)
        this->m_buffers.input_host[i].clear();
      for (size_t i = 0; i < O_; ++i)
        this->m_buffers.output_host[i].clear();
    }

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return ShapeT::flop_count(this->m_dims) * Types::TypeOperations<InputT>::factor * m_batch_count;
    }

    auto number_of_bytes() -> std::optional<size_t> override {
      return ShapeT::byte_count(this->m_dims) * sizeof(InputT) * m_batch_count;
    }

  protected:
    size_t m_batch_count{5};
  };

} // namespace GpuBlas