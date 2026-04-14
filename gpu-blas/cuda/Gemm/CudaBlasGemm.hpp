#pragma once
#include "../CudaBlasWorkload.hpp"
namespace GpuBlas {
  template <typename T>
  struct GemmFunctions;

  template <>
  struct GemmFunctions<float> {
    static constexpr auto standard = cublasSgemm;
    static constexpr auto standard_64 = cublasSgemm_64;
  };

  template <>
  struct GemmFunctions<double> {
    static constexpr auto standard = cublasDgemm;
    static constexpr auto standard_64 = cublasDgemm_64;
  };
  template <>
  struct GemmFunctions<cuComplex> {
    static constexpr auto standard = cublasCgemm;
    static constexpr auto standard_64 = cublasCgemm_64;
  };
  template <>
  struct GemmFunctions<cuDoubleComplex> {
    static constexpr auto standard = cublasZgemm;
    static constexpr auto standard_64 = cublasZgemm_64;
  };
  template <>
  struct GemmFunctions<__half> {
    static constexpr auto standard = cublasHgemm;
    static constexpr auto standard_64 = cublasHgemm_64;
  };

  template <typename T, typename DimType>
  struct GemmSelector {
    static constexpr auto get() {
      if constexpr (std::is_same_v<DimType, std::int64_t>) {
        return GemmFunctions<T>::standard_64;
      } else {
        return GemmFunctions<T>::standard;
      }
    }
  };

  template <typename TypeConfigT, typename DimType>
  class CublasGemm : public CuBlasWorkload<Shapes::GemmShape<TypeConfigT, DimType>> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = CuBlasWorkload<ShapeT>;
    using backend = typename Base::backend;
    void register_options() override {
      CuBlasWorkload<Shapes::GemmShape<TypeConfigT, DimType>>::register_options();
      this->add_option("Gemm", "transA", "The Operation to apply on A", transA);
      this->add_option("Gemm", "transB", "The Operation to apply on B", transB);
    }

  protected:
    cublasOperation_t transA{};
    cublasOperation_t transB{};
  };
  template <typename TypeConfigT, typename DimType>
  class SCublasGemm : public CublasGemm<TypeConfigT, DimType> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = CuBlasWorkload<ShapeT>;
    using backend = typename Base::backend;
    virtual void run_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      using T = typename ShapeT::TypeConfigT::InputT;
      auto gemm_func = GemmSelector<T, DimType>::get();
      CHECK_CUBLAS(cublasSetStream(this->m_handle, *stream));
      CHECK_CUBLAS(gemm_func(this->m_handle, this->transA, this->transB, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A), this->m_dims.m,
                             this->m_buffers.in_device(ShapeT::Inputs::B), this->m_dims.k, &this->m_args.beta,
                             this->m_buffers.out_device(ShapeT::Outputs::C), this->m_dims.m));
    };
  };

} // namespace GpuBlas