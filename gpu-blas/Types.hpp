#ifndef BLAS_BASELINER_TYPES_HPP
#define BLAS_BASELINER_TYPES_HPP
#include <string>
namespace GpuBlas::Types {
  template <typename TypeConfigT>
  auto type_to_string() -> std::string {
    return "";
  };

  template <typename T>
  struct TypeOperations {
    static constexpr size_t factor = 1;
  };

  template <typename T>
  struct ScalarInit {
    static T one() {
      return T{1.0};
    }
    static T zero() {
      return T{0.0};
    }
  };

} // namespace GpuBlas::Types
#endif // BLAS_BASELINER_TYPES_HPP