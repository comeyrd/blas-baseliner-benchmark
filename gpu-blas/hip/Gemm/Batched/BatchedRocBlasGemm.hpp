#pragma once
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/RocBlasWorkload.hpp>
#include <rocblas/rocblas.h>
namespace GpuBlas {
  template <typename T>
  struct BatchedGemmFunctions;

  template <>
  struct BatchedGemmFunctions<float> {
    static constexpr auto standard = rocblas_sgemm_batched;
    static constexpr auto standard_64 = rocblas_sgemm_batched_64;
  };

  template <>
  struct BatchedGemmFunctions<double> {
    static constexpr auto standard = rocblas_dgemm_batched;
    static constexpr auto standard_64 = rocblas_dgemm_batched_64;
  };
  template <>
  struct BatchedGemmFunctions<rocblas_float_complex> {
    static constexpr auto standard = rocblas_cgemm_batched;
    static constexpr auto standard_64 = rocblas_cgemm_batched_64;
  };
  template <>
  struct BatchedGemmFunctions<rocblas_double_complex> {
    static constexpr auto standard = rocblas_zgemm_batched;
    static constexpr auto standard_64 = rocblas_zgemm_batched_64;
  };
  template <>
  struct BatchedGemmFunctions<rocblas_half> {
    static constexpr auto standard = rocblas_hgemm_batched;
    static constexpr auto standard_64 = rocblas_hgemm_batched_64;
  };

  template <typename T, typename DimType>
  struct BatchedGemmSelector {
    static constexpr auto get() {
      if constexpr (std::is_same_v<DimType, std::int64_t>) {
        return BatchedGemmFunctions<T>::standard_64;
      } else {
        return BatchedGemmFunctions<T>::standard;
      }
    }
  };
  template <typename TypeConfigT, typename DimType>
  class BatchedRocBlasGemm : public BatchedCuBlasWorkload<Shapes::GemmShape<TypeConfigT, DimType>> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = BatchedCuBlasWorkload<ShapeT>;
    using backend = typename Base::backend;
    void register_options() override {
      Base::register_options();
      this->add_option("Gemm", "transA", "The Operation to apply on A", transA);
      this->add_option("Gemm", "transB", "The Operation to apply on B", transB);
    }
    virtual std::monostate run_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      using T = typename ShapeT::TypeConfigT::InputT;
      auto gemm_func = BatchedGemmSelector<T, DimType>::get();
      CHECK_ROCBLAS(rocblas_set_stream(this->m_handle, *stream));
      CHECK_ROCBLAS(gemm_func(this->m_handle, transA, transB, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                              &this->m_args.alpha, this->in_device_array(ShapeT::Inputs::A), this->m_dims.m,
                              this->in_device_array(ShapeT::Inputs::B), this->m_dims.k, &this->m_args.beta,
                              this->out_device_array(ShapeT::Outputs::C), this->m_dims.m, this->m_batch_count));

      return {};
    };

  protected:
    rocblas_operation transA{rocblas_operation_none};
    rocblas_operation transB{rocblas_operation_none};
  };
} // namespace GpuBlas