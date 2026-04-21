#ifndef GPU_BLAS_RANDOM_HPP
#define GPU_BLAS_RANDOM_HPP

#include <algorithm>
#include <gpu-blas/Types.hpp>
#include <limits>
#include <random>
#include <vector>

namespace GpuBlas::Random {
  template <typename T>
  struct is_complex : std::false_type {};

  template <typename T>
  struct is_complex<std::complex<T>> : std::true_type {};

  template <typename RefT>
  inline typename std::enable_if<!is_complex<RefT>::value, RefT>::type generate_reference(std::mt19937 &engine) {
    if constexpr (std::is_integral<RefT>::value) {
      std::uniform_int_distribution<RefT> dis(std::numeric_limits<RefT>::lowest() / 4,
                                              std::numeric_limits<RefT>::max() / 4);
      return dis(engine);
    } else {
      std::uniform_real_distribution<RefT> dis(RefT(-1), RefT(1));
      return dis(engine);
    }
  }

  template <typename T>
  inline typename std::enable_if<is_complex<T>::value, T>::type generate_reference(std::mt19937 &engine) {
    using Scalar = typename T::value_type;
    std::uniform_real_distribution<Scalar> dis(Scalar(-1), Scalar(1));
    Scalar real = dis(engine);
    Scalar imag = dis(engine);
    return T(real, imag);
  }

  template <typename TypeT>
  void random_fill(TypeT *data, size_t size, int seed) {
    using RefT = typename Types::ReferenceType<TypeT>::type;

    std::mt19937 engine(seed);

    for (size_t i = 0; i < size; ++i) {
      RefT ref_val = generate_reference<RefT>(engine);
      data[i] = Types::from_reference_type<TypeT>(ref_val);
    }
  }

  template <typename TypeT>
  void random_fill_vector(std::vector<TypeT> &vec, int seed) {
    random_fill(vec.data(), vec.size(), seed);
  }

  enum class FillPolicy {
    None,
    Random,
    Zero
  };

  template <typename TypeT>
  void apply_fill(TypeT *data, size_t size, FillPolicy policy, int seed) {
    switch (policy) {
    case FillPolicy::None:
      break;
    case FillPolicy::Random:
      random_fill(data, size, seed);
      break;
    case FillPolicy::Zero:
      std::fill(data, data + size, Types::ScalarInit<TypeT>::zero());
      break;
    }
  }

  template <typename TypeT>
  void apply_fill(std::vector<TypeT> &buf, FillPolicy policy, int seed) {
    apply_fill(buf.data(), buf.size(), policy, seed);
  }
} // namespace GpuBlas::Random
#endif // GPU_BLAS_RANDOM_HPP