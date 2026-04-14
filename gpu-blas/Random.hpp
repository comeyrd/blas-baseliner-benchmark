#ifndef GPU_BLAS_RANDOM_HPP
#define GPU_BLAS_RANDOM_HPP
#include "Types.hpp"
#include <functional>
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
  void random_fill_vector(std::vector<TypeT> &vec, int seed) {
    using Traits = RandomTraits<TypeT>;
    using S = typename Traits::ScalarT;
    std::mt19937 engine(seed);
    std::uniform_real_distribution<S> dis(std::numeric_limits<S>::lowest() / 4, std::numeric_limits<S>::max() / 4);
    auto source = [&]() { return dis(engine); };
    std::generate(vec.begin(), vec.end(), [&]() { return Traits::create(source); });
  };

  enum class FillPolicy {
    None,
    Random,
    Zero
  };

  template <typename TypeT>
  void apply_fill(std::vector<TypeT> &buf, FillPolicy policy, int seed) {
    switch (policy) {
    case FillPolicy::None:
      break;
    case FillPolicy::Random:
      Random::random_fill_vector(buf, seed);
      break;
    case FillPolicy::Zero:
      std::fill(buf.begin(), buf.end(), Types::ScalarInit<TypeT>::zero());
      break;
    }
  }

} // namespace GpuBlas::Random
#endif // GPU_BLAS_RANDOM_HPP