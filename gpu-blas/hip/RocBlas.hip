#include "RocBlas.hpp"
#include "types.hpp"
#include <baseliner/Conversions.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <rocblas/rocblas.h>

namespace Baseliner::Conversion {

  template <typename XY>
  inline auto xy_to_string(const XY &val) -> std::string {
    return "{" + baseliner_to_string(val.real()) + " ," + baseliner_to_string(val.imag()) + "}";
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
  inline auto baseliner_to_string<rocblas_float_complex>(const rocblas_float_complex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<rocblas_float_complex>(const std::string &val) -> rocblas_float_complex {
    return xy_from_string<rocblas_float_complex>(val);
  };
  template <>
  inline auto baseliner_to_string<rocblas_double_complex>(const rocblas_double_complex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<rocblas_double_complex>(const std::string &val) -> rocblas_double_complex {
    return xy_from_string<rocblas_double_complex>(val);
  };
} // namespace Baseliner::Conversion

namespace GpuBlas {
  template <>
  void HipGemm<float>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_ROCBLAS(rocblas_sgemm(this->handle, rocblas_operation::rocblas_operation_none,
                                rocblas_operation::rocblas_operation_none, this->m_m, this->m_n, this->m_k,
                                &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k, &this->m_beta,
                                this->m_d_C, this->m_m));
  };
  template <>
  void HipGemm<double>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_ROCBLAS(rocblas_dgemm(this->handle, rocblas_operation::rocblas_operation_none,
                                rocblas_operation::rocblas_operation_none, this->m_m, this->m_n, this->m_k,
                                &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k, &this->m_beta,
                                this->m_d_C, this->m_m));
  };
  template <>
  void HipGemm<rocblas_float_complex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_ROCBLAS(rocblas_cgemm(this->handle, rocblas_operation::rocblas_operation_none,
                                rocblas_operation::rocblas_operation_none, this->m_m, this->m_n, this->m_k,
                                &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k, &this->m_beta,
                                this->m_d_C, this->m_m));
  };
  template <>
  void HipGemm<rocblas_double_complex>::run_workload(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_ROCBLAS(rocblas_zgemm(this->handle, rocblas_operation::rocblas_operation_none,
                                rocblas_operation::rocblas_operation_none, this->m_m, this->m_n, this->m_k,
                                &this->m_alpha, this->m_d_A, this->m_m, this->m_d_B, this->m_k, &this->m_beta,
                                this->m_d_C, this->m_m));
  };

  namespace {
    using sgemm = HipGemm<float>;
    using dgemm = HipGemm<double>;
    using cgemm = HipGemm<rocblas_float_complex>;
    using zgemm = HipGemm<rocblas_double_complex>;
    BASELINER_REGISTER_WORKLOAD(sgemm);
    BASELINER_REGISTER_WORKLOAD(dgemm);
    BASELINER_REGISTER_WORKLOAD(cgemm);
    BASELINER_REGISTER_WORKLOAD(zgemm);
  } // namespace

} // namespace GpuBlas
