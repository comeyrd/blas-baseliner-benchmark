#ifndef GPU_BLAS_RANDOM_HPP
#define GPU_BLAS_RANDOM_HPP

#include <algorithm>
#include <gpu-blas/Types.hpp>
#include <limits>
#include <random>
#include <vector>

namespace GpuBlas::Random {

  template <typename T>
  struct RandomTraits {
    using ScalarT = T;
    template <typename Op>
    static T create(Op &&gen) {
      return gen();
    }
  };
  template <typename TypeT>
  void random_fill(TypeT *data, size_t size, int seed) {
    using Traits = RandomTraits<TypeT>;
    using S = typename Traits::ScalarT;
    std::mt19937 engine(seed);
    std::uniform_real_distribution<S> dis(std::numeric_limits<S>::lowest() / 4, std::numeric_limits<S>::max() / 4);
    auto source = [&]() { return dis(engine); };
    for (size_t i = 0; i < size; ++i) {
      data[i] = Traits::create(source);
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