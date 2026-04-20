#pragma once
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/RocBlasWorkload.hpp>
#include <rocblas/rocblas.h>
namespace GpuBlas {

  template <typename T>
  struct StridedBatchedGemmFunctions;

  template <>
  struct StridedBatchedGemmFunctions<float> {
    static constexpr auto standard = rocblas_sgemm_strided_batched;
    static constexpr auto standard_64 = rocblas_sgemm_strided_batched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<double> {
    static constexpr auto standard = rocblas_dgemm_strided_batched;
    static constexpr auto standard_64 = rocblas_dgemm_strided_batched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<rocblas_float_complex> {
    static constexpr auto standard = rocblas_cgemm_strided_batched;
    static constexpr auto standard_64 = rocblas_cgemm_strided_batched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<rocblas_double_complex> {
    static constexpr auto standard = rocblas_zgemm_strided_batched;
    static constexpr auto standard_64 = rocblas_zgemm_strided_batched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<rocblas_half> {
    static constexpr auto standard = rocblas_hgemm_strided_batched;
    static constexpr auto standard_64 = rocblas_hgemm_strided_batched_64;
  };

  template <typename T, typename DimType>
  struct StridedBatchedGemmSelector {
    static constexpr auto get() {
      if constexpr (std::is_same_v<DimType, std::int64_t>) {
        return StridedBatchedGemmFunctions<T>::standard_64;
      } else {
        return StridedBatchedGemmFunctions<T>::standard;
      }
    }
  };

  template <typename TypeConfigT, typename DimType>
  class StridedBatchedRocBlasGemm : public StridedBatchedCuBlasWorkload<Shapes::GemmShape<TypeConfigT, DimType>> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = StridedBatchedCuBlasWorkload<ShapeT>;
    using backend = typename Base::backend;

    void register_options() override {
      Base::register_options();
      this->add_option("Gemm", "transA", "The Operation to apply on A", transA);
      this->add_option("Gemm", "transB", "The Operation to apply on B", transB);
    }

    virtual std::monostate run_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      using T = typename ShapeT::TypeConfigT::InputT;
      auto gemm_func = StridedBatchedGemmSelector<T, DimType>::get();

      CHECK_ROCBLAS(rocblas_set_stream(this->m_handle, *stream));

      // Cast strides to DimType to match the selected API signature (int vs int64_t)
      DimType strideA = static_cast<DimType>(this->get_stride_in(ShapeT::Inputs::A));
      DimType strideB = static_cast<DimType>(this->get_stride_in(ShapeT::Inputs::B));
      DimType strideC = static_cast<DimType>(this->get_stride_out(ShapeT::Outputs::C));

      CHECK_ROCBLAS(gemm_func(this->m_handle, transA, transB, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                              &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A), this->m_dims.m,
                              strideA, this->m_buffers.in_device(ShapeT::Inputs::B), this->m_dims.k, strideB,
                              &this->m_args.beta, this->m_buffers.out_device(ShapeT::Outputs::C), this->m_dims.m,
                              strideC, this->m_batch_count));

      return {};
    };

  protected:
    rocblas_operation transA{rocblas_operation_none};
    rocblas_operation transB{rocblas_operation_none};
  };

} // namespace GpuBlas