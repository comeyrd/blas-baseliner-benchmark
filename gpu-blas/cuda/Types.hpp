#ifndef BLAS_BASELINER_CUDA_TYPES_HPP
#define BLAS_BASELINER_CUDA_TYPES_HPP
#include "cublas_v2.h"
#include "gpu-blas/BlasShapes.hpp"
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
  struct PedanticMath {};
  struct Fast16FMath {};
  struct FastBF16Math {};
  struct FastTF32Math {};
  struct EmulatedBF16x9Math {};

  template <typename TypeT, typename PolicyT>
  struct CublasComputeTraits;

  // --- Half Precision (__half) ---
  template <>
  struct CublasComputeTraits<__half, Types::DefaultMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_16F;
  };
  template <>
  struct CublasComputeTraits<__half, PedanticMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_16F_PEDANTIC;
  };

  // --- Integer (int32_t) ---
  template <>
  struct CublasComputeTraits<int32_t, Types::DefaultMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32I;
  };
  template <>
  struct CublasComputeTraits<int32_t, PedanticMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32I_PEDANTIC;
  };

  // --- Single Precision (float & cuComplex) ---
  // Default Policies
  template <>
  struct CublasComputeTraits<float, Types::DefaultMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F;
  };
  template <>
  struct CublasComputeTraits<cuComplex, Types::DefaultMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F;
  };

  // CUDA-Specific Specialized Policies for FP32
  template <>
  struct CublasComputeTraits<float, PedanticMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F_PEDANTIC;
  };
  template <>
  struct CublasComputeTraits<float, Fast16FMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F_FAST_16F;
  };
  template <>
  struct CublasComputeTraits<float, FastBF16Math> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F_FAST_16BF;
  };
  template <>
  struct CublasComputeTraits<float, FastTF32Math> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F_FAST_TF32;
  };
  // --- Double Precision (double & cuDoubleComplex) ---
  template <>
  struct CublasComputeTraits<double, Types::DefaultMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_64F;
  };
  template <>
  struct CublasComputeTraits<cuDoubleComplex, Types::DefaultMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_64F;
  };
  template <>
  struct CublasComputeTraits<double, PedanticMath> {
    static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_64F_PEDANTIC;
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

    template <>
    struct ReferenceType<__half> {
      using type = float;
      static constexpr double tolerance = 1e-3;
    };

    template <>
    struct ReferenceType<__nv_bfloat16> {
      using type = float;
      static constexpr double tolerance = 1e-3;
    };
    template <>
    struct ReferenceType<cuDoubleComplex> {
      using type = std::complex<double>;
      static constexpr double tolerance = 1e-6;
    };
    template <>
    struct ReferenceType<cuFloatComplex> {
      using type = std::complex<float>;
      static constexpr double tolerance = 1e-6;
    };

    template <>
    inline auto to_reference_type(const cuFloatComplex &val) -> typename ReferenceType<cuFloatComplex>::type {
      return typename ReferenceType<cuFloatComplex>::type(val.x, val.y);
    }

    template <>
    inline auto from_reference_type(const typename ReferenceType<cuFloatComplex>::type &val) -> cuFloatComplex {
      return cuFloatComplex{val.real(), val.imag()};
    }
    template <>
    inline auto to_reference_type(const cuDoubleComplex &val) -> typename ReferenceType<cuDoubleComplex>::type {
      return typename ReferenceType<cuDoubleComplex>::type(val.x, val.y);
    }

    template <>
    inline auto from_reference_type(const typename ReferenceType<cuDoubleComplex>::type &val) -> cuDoubleComplex {
      return cuDoubleComplex{val.real(), val.imag()};
    }

  } // namespace Types
} // namespace GpuBlas

namespace Baseliner::Conversion {
  template <>
  inline auto baseliner_from_string<__half>(const std::string &val) -> __half {
    return GpuBlas::Types::from_string<__half>(val);
  }
  template <>
  inline auto baseliner_to_string<__half>(const __half &val) -> std::string {
    return GpuBlas::Types::to_string(val);
  }
  template <>
  inline auto baseliner_from_string<cuDoubleComplex>(const std::string &val) -> cuDoubleComplex {
    return GpuBlas::Types::from_string<cuDoubleComplex>(val);
  }
  template <>
  inline auto baseliner_to_string<cuDoubleComplex>(const cuDoubleComplex &val) -> std::string {
    return GpuBlas::Types::to_string(val);
  }
  template <>
  inline auto baseliner_from_string<cuFloatComplex>(const std::string &val) -> cuFloatComplex {
    return GpuBlas::Types::from_string<cuFloatComplex>(val);
  }
  template <>
  inline auto baseliner_to_string<cuFloatComplex>(const cuFloatComplex &val) -> std::string {
    return GpuBlas::Types::to_string(val);
  }

  template <>
  inline auto baseliner_from_string<nv_bfloat16>(const std::string &val) -> nv_bfloat16 {
    return GpuBlas::Types::from_string<nv_bfloat16>(val);
  }
  template <>
  inline auto baseliner_to_string<nv_bfloat16>(const nv_bfloat16 &val) -> std::string {
    return GpuBlas::Types::to_string(val);
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
