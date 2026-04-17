#pragma once
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/CudaBlasWorkload.hpp>
namespace GpuBlas {
  template <typename T>
  struct BatchedGemmFunctions;

  template <>
  struct BatchedGemmFunctions<float> {
    static constexpr auto standard = cublasSgemmBatched;
    static constexpr auto standard_64 = cublasSgemmBatched_64;
  };

  template <>
  struct BatchedGemmFunctions<double> {
    static constexpr auto standard = cublasDgemmBatched;
    static constexpr auto standard_64 = cublasDgemmBatched_64;
  };
  template <>
  struct BatchedGemmFunctions<cuComplex> {
    static constexpr auto standard = cublasCgemmBatched;
    static constexpr auto standard_64 = cublasCgemmBatched_64;
  };
  template <>
  struct BatchedGemmFunctions<cuDoubleComplex> {
    static constexpr auto standard = cublasZgemmBatched;
    static constexpr auto standard_64 = cublasZgemmBatched_64;
  };
  template <>
  struct BatchedGemmFunctions<__half> {
    static constexpr auto standard = cublasHgemmBatched;
    static constexpr auto standard_64 = cublasHgemmBatched_64;
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
  class BatchedCublasGemm : public BatchedCuBlasWorkload<Shapes::GemmShape<TypeConfigT, DimType>> {
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
      CHECK_CUBLAS(cublasSetStream(this->m_handle, *stream));
      CHECK_CUBLAS(gemm_func(this->m_handle, transA, transB, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, this->in_device_array(ShapeT::Inputs::A), this->m_dims.m,
                             this->in_device_array(ShapeT::Inputs::B), this->m_dims.k, &this->m_args.beta,
                             this->out_device_array(ShapeT::Outputs::C), this->m_dims.m, this->m_batch_count));

      return {};
    };

  protected:
    cublasOperation_t transA{};
    cublasOperation_t transB{};
  };
} // namespace GpuBlas