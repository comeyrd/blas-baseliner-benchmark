#pragma once
#include "gpu-blas/Types.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gpu-blas/Buffers.hpp>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
namespace GpuBlas::Validation {

  template <typename T>
  inline auto abs_ref(const T &v) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
      return std::abs(v);
    } else {
      return std::fabs(v);
    }
  }

  template <typename ShapeT>
  bool gemm_spot_check(const Buffers<ShapeT> &buffers, const typename ShapeT::DimsT &dims,
                       const typename ShapeT::ArgsT &args, float &mean_error, size_t samples = 128) {

    using InputT = typename ShapeT::InputT;
    using RefT = typename GpuBlas::Types::ReferenceType<InputT>::type;

    constexpr double tol = GpuBlas::Types::ReferenceType<InputT>::tolerance;

    const auto &A = buffers.input_host[ShapeT::Inputs::A];
    const auto &B = buffers.input_host[ShapeT::Inputs::B];
    const auto &C = buffers.output_host[ShapeT::Outputs::C];
    using ScalarT = decltype(std::abs(std::declval<RefT>()));

    size_t m = dims.m;
    size_t n = dims.n;
    size_t k = dims.k;

    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> dist_m(0, m - 1);
    std::uniform_int_distribution<size_t> dist_n(0, n - 1);
    double error{0};
    for (size_t s = 0; s < samples; ++s) {

      size_t i = dist_m(rng);
      size_t j = dist_n(rng);

      RefT dot = RefT{0};

      for (size_t kk = 0; kk < k; ++kk) {
        RefT a = GpuBlas::Types::to_reference_type(A[i + kk * m]);
        RefT b = GpuBlas::Types::to_reference_type(B[kk + j * k]);
        dot += a * b;
      }

      RefT alpha = GpuBlas::Types::to_reference_type(args.alpha);
      RefT beta = GpuBlas::Types::to_reference_type(args.beta);

      RefT c_ref = GpuBlas::Types::to_reference_type(C[i + j * m]);

      RefT expected = alpha * dot; // C is set to 0
      RefT got = c_ref;
      ScalarT abs_err = std::abs(expected - got);
      ScalarT norm_denom = std::max(ScalarT(1), std::abs(expected));
      ScalarT final_scalar_error = abs_err / norm_denom;
      error += final_scalar_error;
    }
    mean_error = error / samples;
    if (mean_error > tol) {
      std::cout << "Warning : mean error on " << samples << " samples " << error / samples << " is above tolerance "
                << tol << "\n";
    }
    return true;
  }

} // namespace GpuBlas::Validation