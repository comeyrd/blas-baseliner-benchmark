#include "../BlasShapes.hpp"
#include "CudaBlasWorkload.hpp"
#include "cublas_v2.h"
#include "types.hpp"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>

namespace GpuBlas {

  using namespace Shapes;

  template <>
  void CublasGemm<float>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    using Shape = GemmShape<TypeConfig<float>>;

    CHECK_CUBLAS(cublasSgemm(this->m_handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, m_buffers.template device_ptr<Shape::A>(), this->m_dims.m,
                             m_buffers.template device_ptr<Shape::B>(), this->m_dims.k, &this->m_args.beta,
                             m_buffers.template device_ptr<Shape::C>(), this->m_dims.m));
  }

  template <>
  void CublasGemm<double>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    using Shape = GemmShape<TypeConfig<double>>;

    CHECK_CUBLAS(cublasDgemm(this->m_handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, m_buffers.template device_ptr<Shape::A>(), this->m_dims.m,
                             m_buffers.template device_ptr<Shape::B>(), this->m_dims.k, &this->m_args.beta,
                             m_buffers.template device_ptr<Shape::C>(), this->m_dims.m));
  }

  template <>
  void CublasGemm<cuComplex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    using Shape = GemmShape<TypeConfig<cuComplex>>;

    CHECK_CUBLAS(cublasCgemm(this->m_handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, m_buffers.template device_ptr<Shape::A>(), this->m_dims.m,
                             m_buffers.template device_ptr<Shape::B>(), this->m_dims.k, &this->m_args.beta,
                             m_buffers.template device_ptr<Shape::C>(), this->m_dims.m));
  }

  template <>
  void CublasGemm<cuDoubleComplex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    using Shape = GemmShape<TypeConfig<cuDoubleComplex>>;

    CHECK_CUBLAS(cublasZgemm(this->m_handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, m_buffers.template device_ptr<Shape::A>(), this->m_dims.m,
                             m_buffers.template device_ptr<Shape::B>(), this->m_dims.k, &this->m_args.beta,
                             m_buffers.template device_ptr<Shape::C>(), this->m_dims.m));
  }

  template <>
  void CublasGemm<__half>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    using Shape = GemmShape<TypeConfig<__half>>;

    CHECK_CUBLAS(cublasHgemm(this->m_handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m_dims.m, this->m_dims.n, this->m_dims.k,
                             &this->m_args.alpha, m_buffers.template device_ptr<Shape::A>(), this->m_dims.m,
                             m_buffers.template device_ptr<Shape::B>(), this->m_dims.k, &this->m_args.beta,
                             m_buffers.template device_ptr<Shape::C>(), this->m_dims.m));
  }

  namespace {
    using sgemm = CublasGemm<float>;
    using dgemm = CublasGemm<double>;
    using cgemm = CublasGemm<cuComplex>;
    using zgemm = CublasGemm<cuDoubleComplex>;
    using hgemm = CublasGemm<__half>;

    BASELINER_REGISTER_WORKLOAD(sgemm);
    BASELINER_REGISTER_WORKLOAD(dgemm);
    BASELINER_REGISTER_WORKLOAD(cgemm);
    BASELINER_REGISTER_WORKLOAD(zgemm);
    BASELINER_REGISTER_WORKLOAD(hgemm);
  } // namespace

} // namespace GpuBlas