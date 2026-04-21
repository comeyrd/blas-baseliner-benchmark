#pragma once
#include "rocblas/rocblas.h"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
#include <gpu-blas/hip/Gemm/RocBlasGemm.hpp>

namespace GpuBlas {
  template <typename TypeConfigT, typename DimType>
  class RocBlasGemmEx : public RocBlasGemm<TypeConfigT, DimType> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = RocBlasWorkload<ShapeT>;
    using backend = typename Base::backend;

    virtual std::monostate run_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      using Config = typename ShapeT::TypeConfigT;
      using InputT = typename Config::InputT;
      using ComputeT = typename Config::ComputeT;
      using OutputT = typename Config::OutputT;

      rocblas_datatype aType = RocblasTypeTraits<InputT>::type;
      rocblas_datatype bType = RocblasTypeTraits<InputT>::type;
      rocblas_datatype cType = RocblasTypeTraits<OutputT>::type;
      rocblas_datatype computeType = RocblasTypeTraits<ComputeT>::type;

      CHECK_ROCBLAS(rocblas_set_stream(this->m_handle, *stream));

      if constexpr (std::is_same_v<DimType, int>) {
        CHECK_ROCBLAS(rocblas_gemm_ex(this->m_handle, this->transA, this->transB, this->m_dims.m, this->m_dims.n,
                                      this->m_dims.k, &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A),
                                      aType, this->m_dims.m, this->m_buffers.in_device(ShapeT::Inputs::B), bType,
                                      this->m_dims.k, &this->m_args.beta,
                                      this->m_buffers.out_device(ShapeT::Outputs::C), cType, this->m_dims.m,
                                      this->m_buffers.out_device(ShapeT::Outputs::C), cType, this->m_dims.m,
                                      computeType, this->algo, this->solution_index, this->flags));
      } else if constexpr (std::is_same_v<DimType, std::int64_t>) {
        CHECK_ROCBLAS(rocblas_gemm_ex_64(this->m_handle, this->transA, this->transB, this->m_dims.m, this->m_dims.n,
                                         this->m_dims.k, &this->m_args.alpha,
                                         this->m_buffers.in_device(ShapeT::Inputs::A), aType, this->m_dims.m,
                                         this->m_buffers.in_device(ShapeT::Inputs::B), bType, this->m_dims.k,
                                         &this->m_args.beta, this->m_buffers.out_device(ShapeT::Outputs::C), cType,
                                         this->m_dims.m, this->m_buffers.out_device(ShapeT::Outputs::C), cType,
                                         this->m_dims.m, computeType, this->algo, this->solution_index, this->flags));
      }
      return {};
    }

    void register_options() override {
      RocBlasGemm<TypeConfigT, DimType>::register_options();
      this->add_option("GemmEx", "algo", "What algo should the GemmEx Use ?", algo);
      this->add_option("GemmEx", "solution_index", "What solution to choose?", solution_index);
      this->add_option("GemmEx", "gemm_flags", "Gemm Flags", flags);
    }

  protected:
    rocblas_gemm_algo algo{rocblas_gemm_algo_standard};
    int32_t solution_index{0};
    uint32_t flags{0};
  };
} // namespace GpuBlas