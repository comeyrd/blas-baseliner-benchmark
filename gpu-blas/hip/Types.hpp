#ifndef BLAS_BASELINER_HIP_TYPES_HPP
#define BLAS_BASELINER_HIP_TYPES_HPP
#include <baseliner/core/AxeSweeping.hpp>
#include <baseliner/core/Conversions.hpp>
#include <gpu-blas/Random.hpp>
#include <gpu-blas/Types.hpp>
#include <hip/hip_runtime.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
/*

typedef enum rocblas_datatype_
{
    rocblas_datatype_f16_r   = 150,< 16-bit floating point, real
    rocblas_datatype_f32_r = 151,      < 32-bit floating point, real
    rocblas_datatype_f64_r = 152,  < 64-bit floating point, real
    rocblas_datatype_f16_c = 153,  < 16-bit floating point, complex
    rocblas_datatype_f32_c = 154,  < 32-bit floating point, complex
    rocblas_datatype_f64_c = 155,  < 64-bit floating point, complex
    rocblas_datatype_i8_r = 160,   <  8-bit signed integer, real
    rocblas_datatype_u8_r = 161,   <  8-bit unsigned integer, real
    rocblas_datatype_i32_r = 162,  < 32-bit signed integer, real
    rocblas_datatype_u32_r = 163,  < 32-bit unsigned integer, real
    rocblas_datatype_i8_c = 164,   <  8-bit signed integer, complex
    rocblas_datatype_u8_c = 165,   <  8-bit unsigned integer, complex
    rocblas_datatype_i32_c = 166,  < 32-bit signed integer, complex
    rocblas_datatype_u32_c = 167,  < 32-bit unsigned integer, complex
    rocblas_datatype_bf16_r = 168, < 16-bit bfloat, real
    rocblas_datatype_bf16_c = 169, < 16-bit bfloat, complex
    rocblas_datatype_invalid = 255,< Invalid datatype value, do not use
}
rocblas_datatype;
*/
namespace GpuBlas {

  template <typename T>
  struct RocblasTypeTraits;

  template <>
  struct RocblasTypeTraits<float> {
    static constexpr rocblas_datatype type = rocblas_datatype_f32_r;
  };
  template <>
  struct RocblasTypeTraits<double> {
    static constexpr rocblas_datatype type = rocblas_datatype_f64_r;
  };
  template <>
  struct RocblasTypeTraits<__half> {
    static constexpr rocblas_datatype type = rocblas_datatype_f16_r;
  };
  template <>
  struct RocblasTypeTraits<__hip_bfloat16> {
    static constexpr rocblas_datatype type = rocblas_datatype_bf16_r;
  };
  template <>
  struct RocblasTypeTraits<rocblas_float_complex> {
    static constexpr rocblas_datatype type = rocblas_datatype_f32_c;
  };
  template <>
  struct RocblasTypeTraits<rocblas_double_complex> {
    static constexpr rocblas_datatype type = rocblas_datatype_f64_c;
  };
  template <>
  struct RocblasTypeTraits<int8_t> {
    static constexpr rocblas_datatype type = rocblas_datatype_i8_r;
  };

  template <>
  struct RocblasTypeTraits<uint8_t> {
    static constexpr rocblas_datatype type = rocblas_datatype_u8_r;
  };

  template <>
  struct RocblasTypeTraits<int32_t> {
    static constexpr rocblas_datatype type = rocblas_datatype_i32_r;
  };

  template <>
  struct RocblasTypeTraits<uint32_t> {
    static constexpr rocblas_datatype type = rocblas_datatype_u32_r;
  };
  namespace Types {
    template <>
    struct ScalarInit<rocblas_float_complex> {
      static rocblas_float_complex one() {
        return rocblas_float_complex{1.0, 1.0};
      }
      static rocblas_float_complex zero() {
        return rocblas_float_complex{0.0, 0.0};
      }
    };
    template <>
    struct ScalarInit<rocblas_double_complex> {
      static rocblas_double_complex one() {
        return rocblas_double_complex{1.0, 1.0};
      }
      static rocblas_double_complex zero() {
        return rocblas_double_complex{0.0, 0.0};
      }
    };
    template <>
    struct ScalarInit<__half> {
      static __half one() {
        return __half{1.0};
      }
      static __half zero() {
        return __half{0.0};
      }
    };
    template <>
    struct ScalarInit<int32_t> {
      static int32_t one() {
        return int32_t{1};
      }
      static int32_t zero() {
        return int32_t{0};
      }
    };
    template <>
    struct ScalarInit<int8_t> {
      static int8_t one() {
        return int8_t{1};
      }
      static int8_t zero() {
        return int8_t{0};
      }
    };

  } // namespace Types
  namespace Random {

    template <>
    struct RandomTraits<rocblas_float_complex> {
      using ScalarT = float;
      template <typename Op>
      static rocblas_float_complex create(Op &&gen) {
        return {gen(), gen()};
      }
    };
    template <>
    struct RandomTraits<rocblas_double_complex> {
      using ScalarT = double;
      template <typename Op>
      static rocblas_double_complex create(Op &&gen) {
        return {gen(), gen()};
      }
    };
    template <>
    struct RandomTraits<__half> {
      using ScalarT = float; // Use float for the random math
      template <typename Op>
      static __half create(Op &&gen) {
        float val = gen();
        return __float2half(val);
      }
    };
    template <>
    struct RandomTraits<__hip_bfloat16> {
      using ScalarT = float; // Use float for the random math
      template <typename Op>
      static __hip_bfloat16 create(Op &&gen) {
        float val = gen();
        return __float2bfloat16(val);
      }
    };
    template <>
    struct RandomTraits<int8_t> {
      using ScalarT = float;
      template <typename Op>
      static int8_t create(Op &&gen) {
        return static_cast<int8_t>(gen());
      }
    };

    template <>
    struct RandomTraits<int32_t> {
      using ScalarT = float;
      template <typename Op>
      static int32_t create(Op &&gen) {
        return static_cast<int32_t>(gen());
      }
    };
  } // namespace Random
} // namespace GpuBlas

namespace Baseliner::Conversion {

  template <typename XY>
  inline auto xy_to_string(const XY &val) -> std::string {
    return "{" + baseliner_to_string(val.real()) + " ," + baseliner_to_string(val.imag()) + "}";
  }
  template <typename XY, typename F>
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

    return {baseliner_from_string<F>(first_part), baseliner_from_string<F>(second_part)};
  }
  template <>
  inline auto baseliner_to_string<rocblas_float_complex>(const rocblas_float_complex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<rocblas_float_complex>(const std::string &val) -> rocblas_float_complex {
    return xy_from_string<rocblas_float_complex, float>(val);
  };
  template <>
  inline auto baseliner_to_string<rocblas_double_complex>(const rocblas_double_complex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<rocblas_double_complex>(const std::string &val) -> rocblas_double_complex {
    return xy_from_string<rocblas_double_complex, double>(val);
  };
  template <>
  inline auto baseliner_from_string<__half>(const std::string &val) -> __half {
    float f_val = std::stof(val);
    return __float2half(f_val);
  }

  template <>
  inline auto baseliner_to_string<__half>(const __half &val) -> std::string {
    float f_val;
    f_val = __half2float(val);
    return std::to_string(f_val);
  }

  template <>
  inline auto baseliner_to_string<rocblas_gemm_algo>(const rocblas_gemm_algo &val) -> std::string {
    switch (val) {
    case rocblas_gemm_algo_standard:
      return "rocblas_gemm_algo_standard";
    case rocblas_gemm_algo_solution_index:
      return "rocblas_gemm_algo_solution_index";
      break;
    };
  }
  template <>
  inline auto baseliner_from_string<rocblas_gemm_algo>(const std::string &val) -> rocblas_gemm_algo {
    if (val == "rocblas_gemm_algo_standard")
      return rocblas_gemm_algo_standard;
    if (val == "rocblas_gemm_algo_solution_index")
      return rocblas_gemm_algo_solution_index;
    return rocblas_gemm_algo_standard;
  }
  template <>
  inline auto baseliner_from_string<rocblas_operation>(const std::string &val) -> rocblas_operation {
    if (val == "rocblas_operation_none")
      return rocblas_operation_none;
    if (val == "rocblas_operation_transpose")
      return rocblas_operation_transpose;
    if (val == "rocblas_operation_conjugate_transpose")
      return rocblas_operation_conjugate_transpose;
    return rocblas_operation_none;
  }

  template <>
  inline auto baseliner_to_string<rocblas_operation>(const rocblas_operation &val) -> std::string {
    switch (val) {
    case rocblas_operation_none:
      return "rocblas_operation_none";
    case rocblas_operation_transpose:
      return "rocblas_operation_transpose";
    case rocblas_operation_conjugate_transpose:
      return "rocblas_operation_conjugate_transpose";
    }
    return "rocblas_operation_none";
  }
} // namespace Baseliner::Conversion

#endif // BLAS_BASELINER_HIP_TYPES_HPP