#include "cublas_v2.h"
#include <gpu-blas/cuda/Gemm/CudaBlasGemm.hpp>
#ifndef BLAS_BASELINER_CudaBlasGemm3m_HPP
#define BLAS_BASELINER_CudaBlasGemm3m_HPP

namespace GpuBlas {
  template <typename T>
  struct Gemm3mFct;

  template <>
  struct Gemm3mFct<cuComplex> {
    static constexpr auto standard = cublasCgemm3m;
    static constexpr auto standard_64 = cublasCgemm3m_64;
  };

  template <>
  struct Gemm3mFct<cuDoubleComplex> {
    static constexpr auto standard = cublasZgemm3m;
    static constexpr auto standard_64 = cublasZgemm3m_64;
  };

  template <typename T, typename DimType>
  struct Gemm3mSelector {
    static constexpr auto get() {
      if constexpr (std::is_same_v<DimType, std::int64_t>) {
        return Gemm3mFct<T>::standard_64;
      } else {
        return Gemm3mFct<T>::standard;
      }
    }
  };

  template <typename TypeConfigT, typename DimType>
  class CublasGemm3m : public CublasGemm<TypeConfigT, DimType> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = CuBlasWorkload<ShapeT>;
    using backend = typename Base::backend;
    virtual std::monostate run_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      using T = typename ShapeT::TypeConfigT::InputT;
      auto gemm_func = Gemm3mSelector<T, DimType>::get();
      CHECK_CUBLAS(cublasSetStream(this->m_handle, *stream));
      CHECK_CUBLAS(gemm_func(this->m_handle, this->transA, this->transB, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A), this->m_dims.m,
                             this->m_buffers.in_device(ShapeT::Inputs::B), this->m_dims.k, &this->m_args.beta,
                             this->m_buffers.out_device(ShapeT::Outputs::C), this->m_dims.m));
      return {};
    };
  };
} // namespace GpuBlas
#endif // BLAS_BASELINER_CudaBlasGemm3m_HPP