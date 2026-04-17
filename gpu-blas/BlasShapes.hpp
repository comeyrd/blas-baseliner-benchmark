#ifndef BLAS_BASELINER_BLASSHAPES_HPP
#define BLAS_BASELINER_BLASSHAPES_HPP
#include <baseliner/core/Options.hpp>
#include <functional>
#include <gpu-blas/Random.hpp>
#include <gpu-blas/Types.hpp>

namespace GpuBlas::Shapes {

  template <typename InputTemplate, typename ComputeTemplate = InputTemplate, typename OutputTemplate = InputTemplate>
  struct TypeConfig {
    using InputT = InputTemplate;
    using ComputeT = ComputeTemplate;
    using OutputT = OutputTemplate;
  };

  // StandardGemm
  template <typename T>
  struct GemmDims : public Baseliner::IOption {
    T m, n, k;

    GemmDims() = default;
    GemmDims(T m_, T n_, T k_)
        : m(m_),
          n(n_),
          k(k_) {};
    void register_options() override {
      this->add_option("GemmDims", "m", "Rows of A and C", m);
      this->add_option("GemmDims", "k", "Cols of A / rows of B", k);
      this->add_option("GemmDims", "n", "Cols of B and C", n);
    }
  };

  template <typename T>
  struct GemmArgs : Baseliner::IOption {
    T alpha = Types::ScalarInit<T>::one();
    T beta = Types::ScalarInit<T>::zero();

    void register_options() override {
      this->add_option("GemmArgs", "alpha", "Scaling factor for A*B", alpha);
      this->add_option("GemmArgs", "beta", "Scaling factor for C", beta);
    }
  };

  template <typename TypeConfigTemplate, typename DimTypes>
  struct GemmShape {
    using TypeConfigT = TypeConfigTemplate;
    using InputT = typename TypeConfigT::InputT;
    using OutputT = typename TypeConfigT::OutputT;
    using ComputeT = typename TypeConfigT::ComputeT;
    enum Inputs : size_t {
      A = 0,
      B = 1,
    };
    enum Outputs : size_t {
      C = 0,
    };

    using DimsT = GemmDims<DimTypes>;
    using ArgsT = GemmArgs<ComputeT>;
    static constexpr std::string_view group = "Gemm";
    static constexpr size_t input_counts = 2;
    static constexpr std::array<Random::FillPolicy, 2> input_fill_policies = {Random::FillPolicy::Random,
                                                                              Random::FillPolicy::Random};
    static constexpr size_t output_counts = 1;
    static constexpr std::array<Random::FillPolicy, 1> output_fill_policies = {Random::FillPolicy::Zero};

    static DimsT scale(size_t work_size) {
      double s = std::pow(static_cast<double>(work_size), 1.0 / 3.0);
      auto snap64 = [](double val) -> size_t {
        size_t v = static_cast<size_t>(val);
        return std::max<size_t>(64, (v / 64) * 64);
      };
      return DimsT(snap64(512 * s), snap64(512 * s), snap64(512 * s));
    };

    static std::array<size_t, 2> input_buffer_sizes(const DimsT &d) {
      return {{static_cast<size_t>(d.m) * d.k, static_cast<size_t>(d.k) * d.n}};
    }

    static std::array<size_t, 1> output_buffer_sizes(const DimsT &d) {
      return {{static_cast<size_t>(d.m) * d.n}};
    }

    static size_t flop_count(const DimsT &d) {
      return 2ULL * d.m * d.n * d.k;
    }
    static size_t byte_count(const DimsT &d) {
      return (d.m * d.k + d.k * d.n) * sizeof(InputT) // A + B
             + (d.m * d.n) * sizeof(OutputT);         // C
    }
  };

} // namespace GpuBlas::Shapes
#endif // BLAS_BASELINER_BLASSHAPES_HPP