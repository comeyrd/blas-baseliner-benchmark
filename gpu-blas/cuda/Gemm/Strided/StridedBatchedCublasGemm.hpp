#pragma once
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/CudaBlasWorkload.hpp>
namespace GpuBlas {

  template <typename T>
  struct StridedBatchedGemmFunctions;

  template <>
  struct StridedBatchedGemmFunctions<float> {
    static constexpr auto standard = cublasSgemmStridedBatched;
    static constexpr auto standard_64 = cublasSgemmStridedBatched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<double> {
    static constexpr auto standard = cublasDgemmStridedBatched;
    static constexpr auto standard_64 = cublasDgemmStridedBatched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<cuComplex> {
    static constexpr auto standard = cublasCgemmStridedBatched;
    static constexpr auto standard_64 = cublasCgemmStridedBatched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<cuDoubleComplex> {
    static constexpr auto standard = cublasZgemmStridedBatched;
    static constexpr auto standard_64 = cublasZgemmStridedBatched_64;
  };

  template <>
  struct StridedBatchedGemmFunctions<__half> {
    static constexpr auto standard = cublasHgemmStridedBatched;
    static constexpr auto standard_64 = cublasHgemmStridedBatched_64;
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
  class StridedBatchedCublasGemm : public StridedBatchedCuBlasWorkload<Shapes::GemmShape<TypeConfigT, DimType>> {
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

      CHECK_CUBLAS(cublasSetStream(this->m_handle, *stream));

      // Cast strides to DimType to match the selected API signature (int vs int64_t)
      DimType strideA = static_cast<DimType>(this->get_stride_in(ShapeT::Inputs::A));
      DimType strideB = static_cast<DimType>(this->get_stride_in(ShapeT::Inputs::B));
      DimType strideC = static_cast<DimType>(this->get_stride_out(ShapeT::Outputs::C));

      CHECK_CUBLAS(gemm_func(this->m_handle, transA, transB, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A), this->m_dims.m, strideA,
                             this->m_buffers.in_device(ShapeT::Inputs::B), this->m_dims.k, strideB, &this->m_args.beta,
                             this->m_buffers.out_device(ShapeT::Outputs::C), this->m_dims.m, strideC,
                             this->m_batch_count));

      return {};
    };

  protected:
    cublasOperation_t transA{CUBLAS_OP_N};
    cublasOperation_t transB{CUBLAS_OP_N};
  };

} // namespace GpuBlas