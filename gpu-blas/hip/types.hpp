#include "../GpuBlas.hpp"
#include "../Random.hpp"
#include <rocblas/rocblas.h>
namespace GpuBlas {
  template <>
  struct TypeOperations<rocblas_float_complex> {
    static constexpr size_t factor = 4;
  };
  template <>
  struct TypeOperations<rocblas_double_complex> {
    static constexpr size_t factor = 4;
  };
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
  } // namespace Random
} // namespace GpuBlas
