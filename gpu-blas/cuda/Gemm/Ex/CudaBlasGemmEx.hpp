#pragma once
#include "cublas_v2.h"
#include <baseliner/Register.hpp>
#include <baseliner/core/Conversions.hpp>
#include <cuda_runtime.h>
#include <gpu-blas/cuda/Gemm/CudaBlasGemm.hpp>

namespace GpuBlas {
  template <typename TypeConfigT, typename DimType>
  class CublasGemmEx : public CublasGemm<TypeConfigT, DimType> {
  public:
    using ShapeT = Shapes::GemmShape<TypeConfigT, DimType>;
    using Base = CuBlasWorkload<ShapeT>;
    using backend = typename Base::backend;

    virtual std::monostate run_workload(std::shared_ptr<typename backend::stream_t> stream) override {
      using Config = typename ShapeT::TypeConfigT;
      using InputT = typename Config::InputT;
      using ComputeT = typename Config::ComputeT;
      using ComputePolicyT = typename Config::ComputePolicyT;
      using OutputT = typename Config::OutputT;

      cudaDataType_t aType = CudaTypeTraits<InputT>::type;
      cudaDataType_t bType = CudaTypeTraits<InputT>::type;
      cudaDataType_t cType = CudaTypeTraits<OutputT>::type;
      cublasComputeType_t computeType = CublasComputeTraits<ComputeT, ComputePolicyT>::type;

      CHECK_CUBLAS(cublasSetStream(this->m_handle, *stream));
      if constexpr (std::is_same_v<DimType, int>) {
        CHECK_CUBLAS(cublasGemmEx(this->m_handle, this->transA, this->transB, this->m_dims.m, this->m_dims.n,
                                  this->m_dims.k, &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A),
                                  aType, this->m_dims.m, this->m_buffers.in_device(ShapeT::Inputs::B), bType,
                                  this->m_dims.k, &this->m_args.beta, this->m_buffers.out_device(ShapeT::Outputs::C),
                                  cType, this->m_dims.m, computeType, this->algo));
      } else if constexpr (std::is_same_v<DimType, std::int64_t>) {
        CHECK_CUBLAS(cublasGemmEx_64(this->m_handle, this->transA, this->transB, this->m_dims.m, this->m_dims.n,
                                     this->m_dims.k, &this->m_args.alpha, this->m_buffers.in_device(ShapeT::Inputs::A),
                                     aType, this->m_dims.m, this->m_buffers.in_device(ShapeT::Inputs::B), bType,
                                     this->m_dims.k, &this->m_args.beta, this->m_buffers.out_device(ShapeT::Outputs::C),
                                     cType, this->m_dims.m, computeType, this->algo));
      }
      return {};
    }

    void register_options() override {
      CublasGemm<TypeConfigT, DimType>::register_options();
      this->add_option("GemmEx", "algo", "What algo should the GemmEx Use ?", algo);
    }

  protected:
    cublasGemmAlgo_t algo;
  };
} // namespace GpuBlas