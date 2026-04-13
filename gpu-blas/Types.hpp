#ifndef BLAS_BASELINER_TYPES_HPP
#define BLAS_BASELINER_TYPES_HPP
#include <string>
namespace GpuBlas::Types {
  template <typename TypeConfigT>
  auto type_to_string() -> std::string;

  template <typename T>
  struct TypeOperations {
    static constexpr size_t factor = 1;
  };

} // namespace GpuBlas::Types
#endif // BLAS_BASELINER_TYPES_HPP