#include "../GpuBlas.hpp"
#include "../Random.hpp"
#include "cublas_v2.h"
namespace GpuBlas {
  template <>
  struct TypeOperations<cuComplex> {
    static constexpr size_t factor = 4;
  };
  template <>
  struct TypeOperations<cuDoubleComplex> {
    static constexpr size_t factor = 4;
  };
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
  } // namespace Random
} // namespace GpuBlas