#ifndef BLAS_BASELINER_HIP_TYPES_HPP
#define BLAS_BASELINER_HIP_TYPES_HPP
#include <baseliner/specs/AxeSweeping.hpp>
#include <baseliner/specs/Conversions.hpp>
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
  struct RocblasTypeTraits<rocblas_half> {
    static constexpr rocblas_datatype type = rocblas_datatype_f16_r;
  };
  template <>
  struct RocblasTypeTraits<rocblas_bfloat16> {
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
    struct ScalarInit<rocblas_half> {
      static rocblas_half one() {
        __half temp{1.0};
        rocblas_half halfval;
        memcpy(&halfval.data, &temp, sizeof(uint16_t));
        return halfval;
      }
      static rocblas_half zero() {
        __half temp{0.0};
        rocblas_half halfval;
        memcpy(&halfval.data, &temp, sizeof(uint16_t));
        return halfval;
      }
    };
    template <>
    struct ScalarInit<rocblas_bfloat16> {
      static rocblas_bfloat16 one() {
        __hip_bfloat16 temp{1.0};
        uint16_t raw = temp;
        rocblas_bfloat16 halfval;
        memcpy(&halfval.data, &raw, sizeof(uint16_t));
        return halfval;
      }
      static rocblas_bfloat16 zero() {
        __hip_bfloat16 temp{0};
        uint16_t raw = temp;
        rocblas_bfloat16 halfval;
        memcpy(&halfval.data, &raw, sizeof(uint16_t));
        return halfval;
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
  namespace Types {
    template <>
    struct ReferenceType<rocblas_half> {
      using type = float;
      static constexpr double tolerance = 1e-3;
    };
    template <>
    inline auto to_reference_type(const rocblas_half &val) -> typename ReferenceType<rocblas_half>::type {
      __half_raw raw;
      raw.x = val.data;
      return static_cast<float>(__half(raw));
    }

    template <>
    inline auto from_reference_type(const typename ReferenceType<rocblas_half>::type &val) -> rocblas_half {
      __half hip_val = __float2half_rn(val);
      rocblas_half ret;
      ret.data = __half_raw(hip_val).x;
      return ret;
    }
    template <>
    struct ReferenceType<rocblas_float_complex> {
      using type = std::complex<float>;
      static constexpr double tolerance = 1e-3;
    };
    template <>
    inline auto to_reference_type(const rocblas_float_complex &val) ->
        typename ReferenceType<rocblas_float_complex>::type {
      return typename ReferenceType<rocblas_float_complex>::type(val.real(), val.imag());
    }

    template <>
    inline auto from_reference_type(const typename ReferenceType<rocblas_float_complex>::type &val)
        -> rocblas_float_complex {
      return rocblas_float_complex{val.real(), val.imag()};
    }
    template <>
    struct ReferenceType<rocblas_double_complex> {
      using type = std::complex<double>;
      static constexpr double tolerance = 1e-3;
    };
    template <>
    inline auto to_reference_type(const rocblas_double_complex &val) ->
        typename ReferenceType<rocblas_double_complex>::type {
      return typename ReferenceType<rocblas_double_complex>::type(val.real(), val.imag());
    }

    template <>
    inline auto from_reference_type(const typename ReferenceType<rocblas_double_complex>::type &val)
        -> rocblas_double_complex {
      return rocblas_double_complex{val.real(), val.imag()};
    }
    template <>
    struct ReferenceType<rocblas_bfloat16> {
      using type = float;
      static constexpr double tolerance = 1e-3;
    };
    template <>
    inline auto to_reference_type(const rocblas_bfloat16 &val) -> typename ReferenceType<rocblas_bfloat16>::type {
      return static_cast<float>(val);
    }

    template <>
    inline auto from_reference_type(const typename ReferenceType<rocblas_bfloat16>::type &val) -> rocblas_bfloat16 {
      return rocblas_bfloat16(val);
    }

  } // namespace Types
} // namespace GpuBlas

namespace Baseliner::Conversion {

  template <>
  inline auto baseliner_to_string<rocblas_half>(const rocblas_half &val) -> std::string {
    return GpuBlas::Types::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<rocblas_bfloat16>(const rocblas_bfloat16 &val) -> std::string {
    return GpuBlas::Types::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<rocblas_double_complex>(const rocblas_double_complex &val) -> std::string {
    return GpuBlas::Types::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<rocblas_float_complex>(const rocblas_float_complex &val) -> std::string {
    return GpuBlas::Types::to_string(val);
  }
  template <>
  inline auto baseliner_from_string<rocblas_half>(const std::string &val) -> rocblas_half {
    return GpuBlas::Types::from_string<rocblas_half>(val);
  }
  template <>
  inline auto baseliner_from_string<rocblas_bfloat16>(const std::string &val) -> rocblas_bfloat16 {
    return GpuBlas::Types::from_string<rocblas_bfloat16>(val);
  }
  template <>
  inline auto baseliner_from_string<rocblas_double_complex>(const std::string &val) -> rocblas_double_complex {
    return GpuBlas::Types::from_string<rocblas_double_complex>(val);
  }
  template <>
  inline auto baseliner_from_string<rocblas_float_complex>(const std::string &val) -> rocblas_float_complex {
    return GpuBlas::Types::from_string<rocblas_float_complex>(val);
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