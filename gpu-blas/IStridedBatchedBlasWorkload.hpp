#pragma once
#include <gpu-blas/IBlasWorkload.hpp>

namespace GpuBlas {

  template <typename BackendT, typename ShapeT, typename MemoryBackendT>
  class IStridedBatchedBlasWorkload : public IBlasWorkload<BackendT, ShapeT, MemoryBackendT> {
  public:
    using Base = IBlasWorkload<BackendT, ShapeT, MemoryBackendT>;
    using InputT = typename ShapeT::InputT;
    using OutputT = typename ShapeT::OutputT;

    static constexpr size_t I_ = ShapeT::input_counts;
    static constexpr size_t O_ = ShapeT::output_counts;

    void register_options() override {
      Base::register_options();
      this->add_option("Batched", "batch_count", "Number of batches", m_batch_count);
    }

    void alloc_host() override {
      this->m_dims = ShapeT::scale(this->get_work_size());

      auto input_sz = ShapeT::input_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < I_; ++i) {
        this->m_buffers.input_host[i].resize(input_sz[i] * m_batch_count);
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
          Random::apply_fill(ptr, slice, ShapeT::output_fill_policies[i], this->get_seed() + b);
        }
      }
    }

    void setup(std::shared_ptr<typename BackendT::stream_t> stream) override {
      this->alloc_host();
      this->alloc_handle();

      auto input_sz = ShapeT::input_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < I_; ++i) {
        this->m_memory.malloc(&this->m_buffers.input_device[i], input_sz[i] * m_batch_count * sizeof(InputT), stream);
        this->m_memory.memcpy_to_device(this->m_buffers.input_device[i], this->m_buffers.input_host[i].data(),
                                        input_sz[i] * m_batch_count * sizeof(InputT), stream);
      }

      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < O_; i++) {
        this->m_memory.malloc(&this->m_buffers.output_device[i], output_sz[i] * m_batch_count * sizeof(OutputT),
                              stream);
        this->m_memory.memcpy_to_device(this->m_buffers.output_device[i], this->m_buffers.output_host[i].data(),
                                        output_sz[i] * m_batch_count * sizeof(OutputT), stream);
      }
    }

    void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) override {
      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);
      for (size_t i = 0; i < O_; i++) {
        this->m_memory.memset(this->m_buffers.output_device[i], 0, output_sz[i] * m_batch_count * sizeof(OutputT),
                              stream);
      }
    }

    void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override {
      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);

      for (size_t i = 0; i < I_; ++i) {
        this->m_memory.free(this->m_buffers.input_device[i]);
      }
      for (size_t i = 0; i < O_; i++) {
        this->m_memory.memcpy_to_host(this->m_buffers.output_host[i].data(), this->m_buffers.output_device[i],
                                      output_sz[i] * m_batch_count * sizeof(OutputT), stream);
      }
      BackendT::synchronize(stream);
      for (size_t i = 0; i < O_; i++) {
        this->m_memory.free(this->m_buffers.output_device[i]);
      }
      this->free_handle();
      this->inner_validate();
      this->free_host();
    }

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return ShapeT::flop_count(this->m_dims) * Types::TypeOperations<InputT>::factor * m_batch_count;
    }

    auto number_of_bytes() -> std::optional<size_t> override {
      return ShapeT::byte_count(this->m_dims) * sizeof(InputT) * m_batch_count;
    }

    auto get_stride_in(size_t i) const -> long long int {
      auto input_sz = ShapeT::input_buffer_sizes(this->m_dims);
      return static_cast<long long int>(input_sz[i]);
    }

    auto get_stride_out(size_t i) const -> long long int {
      auto output_sz = ShapeT::output_buffer_sizes(this->m_dims);
      return static_cast<long long int>(output_sz[i]);
    }
    virtual void inner_validate() override {
      this->m_valid = true;

      for (size_t i = 0; i < O_; i++) {
        auto &out = this->m_buffers.output_host[i];
        if (out.empty())
          continue;

        const uint8_t *out_ptr = reinterpret_cast<const uint8_t *>(out.data());
        size_t out_bytes = out.size() * sizeof(OutputT);
        size_t check_bytes = std::min(out_bytes, size_t(256));

        uint64_t out_hash = 0;
        for (size_t j = 0; j < check_bytes; ++j) {
          out_hash += out_ptr[j];
        }

        if (out_hash == 0) {
          this->m_valid = false;
          std::cout << "Wall of zeros\n";
          return;
        }

        for (size_t j = 0; j < I_; ++j) {
          auto &in = this->m_buffers.input_host[j];
          const uint8_t *in_ptr = reinterpret_cast<const uint8_t *>(in.data());
          size_t in_bytes = in.size() * sizeof(InputT);

          if (in_bytes < check_bytes)
            continue;

          uint64_t in_hash = 0;
          for (size_t k = 0; k < check_bytes; ++k) {
            in_hash += in_ptr[k];
          }

          if (out_hash == in_hash) {
            std::cout << "here\n";
            this->m_valid = false;
            return;
          }
        }
      }
    }

  protected:
    size_t m_batch_count{5};
  };

} // namespace GpuBlas