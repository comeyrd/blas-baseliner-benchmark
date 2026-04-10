#include "CudaBlas.hpp"
#include "cublas_v2.h"
#include "types.hpp"
#include <baseliner/core/Conversions.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>

namespace Baseliner::Sweep::Detail {
  template <>
  struct Sweeper<float2> {
    static auto generate(const TypedSweepHint<float2> &hint) -> std::vector<float2> {
      if (hint.m_policy == SweepPolicy::Enumerated) {
        return hint.m_enumerated;
      }

      std::vector<float2> result;

      if (hint.m_policy == SweepPolicy::LinearRange) {
        // Basic validation for float2 steps
        if (hint.m_step.x <= 0.0f || hint.m_step.y <= 0.0f) {
          throw std::invalid_argument("Step components must be > 0");
        }

        // Nested loops to generate the 2D grid
        for (float x = hint.m_min.x; x <= hint.m_max.x; x += hint.m_step.x) {
          for (float y = hint.m_min.y; y <= hint.m_max.y; y += hint.m_step.y) {
            result.push_back({x, y});
          }
        }
      } else if (hint.m_policy == SweepPolicy::PowersOfTwo) {
        if (hint.m_min.x <= 0.0f || hint.m_min.y <= 0.0f) {
          throw std::invalid_argument("Min components must be > 0 for PowersOfTwo");
        }

        for (float x = hint.m_min.x; x <= hint.m_max.x; x *= 2.0f) {
          for (float y = hint.m_min.y; y <= hint.m_max.y; y *= 2.0f) {
            result.push_back({x, y});
          }
        }
      }

      return result;
    }
  };
  template <>
  struct Sweeper<double2> {
    static auto generate(const TypedSweepHint<double2> &hint) -> std::vector<double2> {
      if (hint.m_policy == SweepPolicy::Enumerated)
        return hint.m_enumerated;
      std::vector<double2> result;
      if (hint.m_policy == SweepPolicy::LinearRange) {
        for (double x = hint.m_min.x; x <= hint.m_max.x; x += hint.m_step.x) {
          for (double y = hint.m_min.y; y <= hint.m_max.y; y += hint.m_step.y) {
            result.push_back({x, y});
          }
        }
      }
      // Add PowersOfTwo if needed, following the same nested loop logic
      return result;
    }
  };

} // namespace Baseliner::Sweep::Detail
namespace Baseliner::Conversion {

  template <typename XY>
  inline auto xy_to_string(const XY &val) -> std::string {
    return "{" + baseliner_to_string(val.x) + " ," + baseliner_to_string(val.y) + "}";
  }
  template <typename XY>
  auto xy_from_string(const std::string &val) -> XY {
    std::string string_v = trim_before_after_whitespace(val);

    if (string_v.size() >= 2 && ((string_v.front() == '{' && string_v.back() == '}'))) {
      string_v = string_v.substr(1, string_v.size() - 2);
    }
    size_t comma_pos = string_v.find(',');
    if (comma_pos == std::string::npos) {
      throw std::invalid_argument("Input does not contain a pair separator: " + val);
    }
    std::string first_part = trim_before_after_whitespace(string_v.substr(0, comma_pos));
    std::string second_part = trim_before_after_whitespace(string_v.substr(comma_pos + 1));

    return {baseliner_from_string<float>(first_part), baseliner_from_string<float>(second_part)};
  }
  template <>
  inline auto baseliner_to_string<float2>(const float2 &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<float2>(const std::string &val) -> float2 {
    return xy_from_string<float2>(val);
  };
  template <>
  inline auto baseliner_to_string<double2>(const double2 &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<double2>(const std::string &val) -> double2 {
    return xy_from_string<double2>(val);
  };
} // namespace Baseliner::Conversion

namespace GpuBlas {
  template <>
  void CudaGemm<float>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasSgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  template <>
  void CudaGemm<double>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasDgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  template <>
  void CudaGemm<cuComplex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasCgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };
  template <>
  void CudaGemm<cuDoubleComplex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUBLAS(cublasZgemm(this->handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, this->m_m,
                             this->m_n, this->m_k, &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k,
                             &this->m_beta, this->m_d_C, this->m_m));
  };

  namespace {
    using sgemm = CudaGemm<float>;
    using dgemm = CudaGemm<double>;
    using cgemm = CudaGemm<cuComplex>;
    using zgemm = CudaGemm<cuDoubleComplex>;
    BASELINER_REGISTER_WORKLOAD(sgemm);
    BASELINER_REGISTER_WORKLOAD(dgemm);
    BASELINER_REGISTER_WORKLOAD(cgemm);
    BASELINER_REGISTER_WORKLOAD(zgemm);
  } // namespace

} // namespace GpuBlas