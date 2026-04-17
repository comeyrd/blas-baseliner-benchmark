#pragma once

#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/cuda/CudaBlasWorkload.hpp>
namespace GpuBlas {
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

} // namespace GpuBlas