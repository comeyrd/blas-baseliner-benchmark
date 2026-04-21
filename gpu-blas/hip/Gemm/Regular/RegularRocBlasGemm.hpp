#pragma once
#include "rocblas/rocblas.h"
#include <cstdint>
#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/Gemm/RocBlasGemm.hpp>
#include <gpu-blas/hip/RocBlasWorkload.hpp>

namespace GpuBlas {
  template <typename T>
  struct GemmFunctions;

  template <>
  struct GemmFunctions<float> {
    static constexpr auto standard = rocblas_sgemm;
    static constexpr auto standard_64 = rocblas_sgemm_64;
  };

  template <>
  struct GemmFunctions<double> {
    static constexpr auto standard = rocblas_dgemm;
    static constexpr auto standard_64 = rocblas_dgemm_64;
  };
  template <>
  struct GemmFunctions<rocblas_float_complex> {
    static constexpr auto standard = rocblas_cgemm;
    static constexpr auto standard_64 = rocblas_cgemm_64;
  };
  template <>
  struct GemmFunctions<rocblas_double_complex> {
    static constexpr auto standard = rocblas_zgemm;
    static constexpr auto standard_64 = rocblas_zgemm_64;
  };
  template <>
  struct GemmFunctions<rocblas_half> {
    static constexpr auto standard = rocblas_hgemm;
    static constexpr auto standard_64 = rocblas_hgemm_64;
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
  class RegularRocBlasGemm : public RocBlasGemm<TypeConfigT, DimType> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = RocBlasWorkload<ShapeT>;
    using backend = typename Base::backend;
    virtual std::monostate run_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      using T = typename ShapeT::TypeConfigT::InputT;
      auto gemm_func = GemmSelector<T, DimType>::get();
      DimType lda = (this->transA == rocblas_operation_none) ? this->m_dims.m : this->m_dims.k;
      DimType ldb = (this->transB == rocblas_operation_none) ? this->m_dims.k : this->m_dims.n;
      DimType ldc = this->m_dims.m;
      CHECK_ROCBLAS(rocblas_set_stream(this->m_handle, *stream));
      CHECK_ROCBLAS(gemm_func(this->m_handle, this->transA, this->transB, this->m_dims.m, this->m_dims.n,
                              this->m_dims.k, &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A), lda,
                              this->m_buffers.in_device(ShapeT::Inputs::B), ldb, &this->m_args.beta,
                              this->m_buffers.out_device(ShapeT::Outputs::C), ldc));
      return {};
    };
  };

} // namespace GpuBlas