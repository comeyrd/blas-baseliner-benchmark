#pragma once

#include <gpu-blas/BlasShapes.hpp>
#include <gpu-blas/hip/RocBlasWorkload.hpp>
namespace GpuBlas {
  template <typename TypeConfigT, typename DimType>
  class RocBlasGemm : public CuBlasWorkload<Shapes::GemmShape<TypeConfigT, DimType>> {
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
    rocblas_operation transA{rocblas_operation::rocblas_operation_none};
    rocblas_operation transB{rocblas_operation::rocblas_operation_none};
  };

} // namespace GpuBlas