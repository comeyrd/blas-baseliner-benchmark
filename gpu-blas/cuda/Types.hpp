#ifndef BLAS_BASELINER_CUDA_TYPES_HPP
#define BLAS_BASELINER_CUDA_TYPES_HPP
#include "cublas_v2.h"
#include <baseliner/core/AxeSweeping.hpp>
#include <baseliner/core/Conversions.hpp>
#include <gpu-blas/Random.hpp>
#include <gpu-blas/Types.hpp>

namespace GpuBlas {

  template <typename T>
  struct CudaTypeTraits;
  template <>
  struct CudaTypeTraits<float> {
    static constexpr cudaDataType_t type = CUDA_R_32F;
  };
  template <>
  struct CudaTypeTraits<double> {
    static constexpr cudaDataType_t type = CUDA_R_64F;
  };
  template <>
  struct CudaTypeTraits<__half> {
    static constexpr cudaDataType_t type = CUDA_R_16F;
  };
  template <>
  struct CudaTypeTraits<cuComplex> {
    static constexpr cudaDataType_t type = CUDA_C_32F;
  };
  template <>
  struct CudaTypeTraits<cuDoubleComplex> {
    static constexpr cudaDataType_t type = CUDA_C_64F;
  };
  template <>
  struct CudaTypeTraits<__nv_bfloat16> {
    static constexpr cudaDataType_t type = CUDA_R_16BF;
  };

  template <>
  struct CudaTypeTraits<int8_t> {
    static constexpr cudaDataType_t type = CUDA_R_8I;
  };
  template <>
  struct CudaTypeTraits<int32_t> {
    static constexpr cudaDataType_t type = CUDA_R_32I;
  };
  template <typename T>
  struct CublasComputeTraits;
  template <>
  struct CublasComputeTraits<float> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F;
  };
  template <>
  struct CublasComputeTraits<double> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_64F;
  };
  template <>
  struct CublasComputeTraits<__half> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_16F;
  };
  template <>
  struct CublasComputeTraits<int32_t> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32I;
  };
  namespace Types {
    template <>
    struct TypeOperations<cuComplex> {
      static constexpr size_t factor = 4;
    };
    template <>
    struct TypeOperations<cuDoubleComplex> {
      static constexpr size_t factor = 4;
    };

    template <>
    struct ScalarInit<float2> {
      static float2 one() {
        return float2{1.0, 1.0};
      }
      static float2 zero() {
        return float2{0.0, 0.0};
      }
    };
    template <>
    struct ScalarInit<double2> {
      static double2 one() {
        return double2{1.0, 1.0};
      }
      static double2 zero() {
        return double2{0.0, 0.0};
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
    struct RandomTraits<cuComplex> {
      using ScalarT = float;
      template <typename Op>
      static cuComplex create(Op &&gen) {
        return {gen(), gen()};
      }
    };
    template <>
    struct RandomTraits<cuDoubleComplex> {
      using ScalarT = double;
      template <typename Op>
      static cuDoubleComplex create(Op &&gen) {
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
    struct RandomTraits<__nv_bfloat16> {
      using ScalarT = float; // Use float for the random math
      template <typename Op>
      static __nv_bfloat16 create(Op &&gen) {
        float val = gen();
        return __float2bfloat16(val);
      }
    };
    template <>
    struct RandomTraits<int8_t> {
      using ScalarT = float; // MUST be a floating point type for std::uniform_real_distribution
      template <typename Op>
      static int8_t create(Op &&gen) {
        return static_cast<int8_t>(gen());
      }
    };

    template <>
    struct RandomTraits<int32_t> {
      using ScalarT = float; // MUST be a floating point type
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
    return "{" + baseliner_to_string(val.x) + " ," + baseliner_to_string(val.y) + "}";
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
  inline auto baseliner_to_string<cuComplex>(const cuComplex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<cuComplex>(const std::string &val) -> cuComplex {
    return xy_from_string<cuComplex, float>(val);
  };
  template <>
  inline auto baseliner_to_string<cuDoubleComplex>(const cuDoubleComplex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<cuDoubleComplex>(const std::string &val) -> cuDoubleComplex {
    return xy_from_string<cuDoubleComplex, double>(val);
  };
  template <>
  inline auto baseliner_from_string<__half>(const std::string &val) -> __half {
    float f_val = std::stof(val);
    return __half{__float2half(f_val)};
  }

  template <>
  inline auto baseliner_to_string<__half>(const __half &val) -> std::string {
    float f_val;
    f_val = __half2float(val);
    return std::to_string(f_val);
  }
  template <>
  inline auto baseliner_to_string<cublasGemmAlgo_t>(const cublasGemmAlgo_t &val) -> std::string {
    switch (val) {
    case CUBLAS_GEMM_DEFAULT:
      return "CUBLAS_GEMM_DEFAULT";

    case CUBLAS_GEMM_ALGO0:
      return "CUBLAS_GEMM_ALGO0";
    case CUBLAS_GEMM_ALGO1:
      return "CUBLAS_GEMM_ALGO1";
    case CUBLAS_GEMM_ALGO2:
      return "CUBLAS_GEMM_ALGO2";
    case CUBLAS_GEMM_ALGO3:
      return "CUBLAS_GEMM_ALGO3";
    case CUBLAS_GEMM_ALGO4:
      return "CUBLAS_GEMM_ALGO4";
    case CUBLAS_GEMM_ALGO5:
      return "CUBLAS_GEMM_ALGO5";
    case CUBLAS_GEMM_ALGO6:
      return "CUBLAS_GEMM_ALGO6";
    case CUBLAS_GEMM_ALGO7:
      return "CUBLAS_GEMM_ALGO7";
    case CUBLAS_GEMM_ALGO8:
      return "CUBLAS_GEMM_ALGO8";
    case CUBLAS_GEMM_ALGO9:
      return "CUBLAS_GEMM_ALGO9";
    case CUBLAS_GEMM_ALGO10:
      return "CUBLAS_GEMM_ALGO10";
    case CUBLAS_GEMM_ALGO11:
      return "CUBLAS_GEMM_ALGO11";
    case CUBLAS_GEMM_ALGO12:
      return "CUBLAS_GEMM_ALGO12";
    case CUBLAS_GEMM_ALGO13:
      return "CUBLAS_GEMM_ALGO13";
    case CUBLAS_GEMM_ALGO14:
      return "CUBLAS_GEMM_ALGO14";
    case CUBLAS_GEMM_ALGO15:
      return "CUBLAS_GEMM_ALGO15";

    case CUBLAS_GEMM_DEFAULT_TENSOR_OP:
      return "CUBLAS_GEMM_DEFAULT_TENSOR_OP";

    case CUBLAS_GEMM_ALGO0_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO0_TENSOR_OP";
    case CUBLAS_GEMM_ALGO1_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO1_TENSOR_OP";
    case CUBLAS_GEMM_ALGO2_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO2_TENSOR_OP";
    case CUBLAS_GEMM_ALGO3_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO3_TENSOR_OP";
    case CUBLAS_GEMM_ALGO4_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO4_TENSOR_OP";
    case CUBLAS_GEMM_ALGO5_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO5_TENSOR_OP";
    case CUBLAS_GEMM_ALGO6_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO6_TENSOR_OP";
    case CUBLAS_GEMM_ALGO7_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO7_TENSOR_OP";
    case CUBLAS_GEMM_ALGO8_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO8_TENSOR_OP";
    case CUBLAS_GEMM_ALGO9_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO9_TENSOR_OP";
    case CUBLAS_GEMM_ALGO10_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO10_TENSOR_OP";
    case CUBLAS_GEMM_ALGO11_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO11_TENSOR_OP";
    case CUBLAS_GEMM_ALGO12_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO12_TENSOR_OP";
    case CUBLAS_GEMM_ALGO13_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO13_TENSOR_OP";
    case CUBLAS_GEMM_ALGO14_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO14_TENSOR_OP";
    case CUBLAS_GEMM_ALGO15_TENSOR_OP:
      return "CUBLAS_GEMM_ALGO15_TENSOR_OP";
    case CUBLAS_GEMM_ALGO16:
      return "CUBLAS_GEMM_ALGO16";
    case CUBLAS_GEMM_ALGO17:
      return "CUBLAS_GEMM_ALGO17";
    case CUBLAS_GEMM_ALGO18:
      return "CUBLAS_GEMM_ALGO18";
    case CUBLAS_GEMM_ALGO19:
      return "CUBLAS_GEMM_ALGO19";
    case CUBLAS_GEMM_ALGO20:
      return "CUBLAS_GEMM_ALGO20";
    case CUBLAS_GEMM_ALGO21:
      return "CUBLAS_GEMM_ALGO21";
    case CUBLAS_GEMM_ALGO22:
      return "CUBLAS_GEMM_ALGO22";
    case CUBLAS_GEMM_ALGO23:
      return "CUBLAS_GEMM_ALGO23";
      break;
    }

    return "UNKNOWN_CUBLAS_GEMM_ALGO";
  }
  template <>
  inline auto baseliner_from_string<cublasGemmAlgo_t>(const std::string &val) -> cublasGemmAlgo_t {
    if (val == "CUBLAS_GEMM_DFALT" || val == "CUBLAS_GEMM_DEFAULT")
      return CUBLAS_GEMM_DEFAULT;
    if (val == "CUBLAS_GEMM_ALGO0")
      return CUBLAS_GEMM_ALGO0;
    if (val == "CUBLAS_GEMM_ALGO1")
      return CUBLAS_GEMM_ALGO1;
    if (val == "CUBLAS_GEMM_ALGO2")
      return CUBLAS_GEMM_ALGO2;
    if (val == "CUBLAS_GEMM_ALGO3")
      return CUBLAS_GEMM_ALGO3;
    if (val == "CUBLAS_GEMM_ALGO4")
      return CUBLAS_GEMM_ALGO4;
    if (val == "CUBLAS_GEMM_ALGO5")
      return CUBLAS_GEMM_ALGO5;
    if (val == "CUBLAS_GEMM_ALGO6")
      return CUBLAS_GEMM_ALGO6;
    if (val == "CUBLAS_GEMM_ALGO7")
      return CUBLAS_GEMM_ALGO7;
    if (val == "CUBLAS_GEMM_ALGO8")
      return CUBLAS_GEMM_ALGO8;
    if (val == "CUBLAS_GEMM_ALGO9")
      return CUBLAS_GEMM_ALGO9;
    if (val == "CUBLAS_GEMM_ALGO10")
      return CUBLAS_GEMM_ALGO10;
    if (val == "CUBLAS_GEMM_ALGO11")
      return CUBLAS_GEMM_ALGO11;
    if (val == "CUBLAS_GEMM_ALGO12")
      return CUBLAS_GEMM_ALGO12;
    if (val == "CUBLAS_GEMM_ALGO13")
      return CUBLAS_GEMM_ALGO13;
    if (val == "CUBLAS_GEMM_ALGO14")
      return CUBLAS_GEMM_ALGO14;
    if (val == "CUBLAS_GEMM_ALGO15")
      return CUBLAS_GEMM_ALGO15;
    if (val == "CUBLAS_GEMM_ALGO16")
      return CUBLAS_GEMM_ALGO16;
    if (val == "CUBLAS_GEMM_ALGO17")
      return CUBLAS_GEMM_ALGO17;
    if (val == "CUBLAS_GEMM_ALGO18")
      return CUBLAS_GEMM_ALGO18;
    if (val == "CUBLAS_GEMM_ALGO19")
      return CUBLAS_GEMM_ALGO19;
    if (val == "CUBLAS_GEMM_ALGO20")
      return CUBLAS_GEMM_ALGO20;
    if (val == "CUBLAS_GEMM_ALGO21")
      return CUBLAS_GEMM_ALGO21;
    if (val == "CUBLAS_GEMM_ALGO22")
      return CUBLAS_GEMM_ALGO22;
    if (val == "CUBLAS_GEMM_ALGO23")
      return CUBLAS_GEMM_ALGO23;

    if (val == "CUBLAS_GEMM_DEFAULT_TENSOR_OP" || val == "CUBLAS_GEMM_DFALT_TENSOR_OP")
      return CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    if (val == "CUBLAS_GEMM_ALGO0_TENSOR_OP")
      return CUBLAS_GEMM_ALGO0_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO1_TENSOR_OP")
      return CUBLAS_GEMM_ALGO1_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO2_TENSOR_OP")
      return CUBLAS_GEMM_ALGO2_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO3_TENSOR_OP")
      return CUBLAS_GEMM_ALGO3_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO4_TENSOR_OP")
      return CUBLAS_GEMM_ALGO4_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO5_TENSOR_OP")
      return CUBLAS_GEMM_ALGO5_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO6_TENSOR_OP")
      return CUBLAS_GEMM_ALGO6_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO7_TENSOR_OP")
      return CUBLAS_GEMM_ALGO7_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO8_TENSOR_OP")
      return CUBLAS_GEMM_ALGO8_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO9_TENSOR_OP")
      return CUBLAS_GEMM_ALGO9_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO10_TENSOR_OP")
      return CUBLAS_GEMM_ALGO10_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO11_TENSOR_OP")
      return CUBLAS_GEMM_ALGO11_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO12_TENSOR_OP")
      return CUBLAS_GEMM_ALGO12_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO13_TENSOR_OP")
      return CUBLAS_GEMM_ALGO13_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO14_TENSOR_OP")
      return CUBLAS_GEMM_ALGO14_TENSOR_OP;
    if (val == "CUBLAS_GEMM_ALGO15_TENSOR_OP")
      return CUBLAS_GEMM_ALGO15_TENSOR_OP;

    throw std::invalid_argument("Unknown cublasGemmAlgo_t: " + val);
  }
  template <>
  inline auto baseliner_from_string<cublasOperation_t>(const std::string &val) -> cublasOperation_t {
    if (val == "CUBLAS_OP_N")
      return CUBLAS_OP_N;
    if (val == "CUBLAS_OP_T")
      return CUBLAS_OP_T;
    if (val == "CUBLAS_OP_C")
      return CUBLAS_OP_C;
    if (val == "CUBLAS_OP_HERMITAN")
      return CUBLAS_OP_HERMITAN;
    if (val == "CUBLAS_OP_CONJG")
      return CUBLAS_OP_CONJG;
    throw std::invalid_argument("Unknown cublasOperation_t: " + val);
  }

  template <>
  inline auto baseliner_to_string<cublasOperation_t>(const cublasOperation_t &val) -> std::string {
    switch (val) {
    case CUBLAS_OP_N:
      return "CUBLAS_OP_N";
    case CUBLAS_OP_T:
      return "CUBLAS_OP_T";
    case CUBLAS_OP_HERMITAN:
      return "CUBLAS_OP_HERMITAN";
    case CUBLAS_OP_CONJG:
      return "CUBLAS_OP_CONJG";
    }
    return "UNKNOWN_CUBLAS_OPERATION";
  }
} // namespace Baseliner::Conversion

#endif // BLAS_BASELINER_CUDA_TYPES_HPP
