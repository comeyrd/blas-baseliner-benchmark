#ifndef BLAS_BASELINER_BLASSHAPES_HPP
#define BLAS_BASELINER_BLASSHAPES_HPP
#include "Random.hpp"
#include <vector>
namespace GpuBlas::Shapes {

  template <typename TypeT>
  struct BufferSlot {
    std::vector<TypeT> data;
    Random::FillPolicy fill;
    bool is_output; // Is output set to true : copied back at teardown & reset to 0
  };

  struct GemmDims {
    size_t m, n, k;
  };

  template <typename T>
  struct GemmArgs {
    T alpha{static_cast<T>(1)};
    T beta{static_cast<T>(0)};

    template <typename WorkloadT>
    void register_options(WorkloadT &w) {
      w.add_option("Gemm", "alpha", "Scaling factor for A*B", alpha);
      w.add_option("Gemm", "beta", "Scaling factor for C", beta);
    }
  };

  template <typename InputTemplate, typename ComputeTemlplate = InputTemplate, typename OutputTemplate = InputTemplate>
  struct TypeConfig {
    using InputT = InputTemplate;
    using ComputeT = ComputeTemlplate;
    using OutputT = OutputTemplate;
  };

  template <typename TypeConfigT>
  struct GemmShape {
    using InputT = typename TypeConfigT::InputT;
    using OutputT = typename TypeConfigT::OutputT;
    using ComputeT = typename TypeConfigT::ComputeT;
    enum Slot : size_t {
      A = 0,
      B = 1,
      C = 2
    };

    template <size_t I>
    using BufferT = std::conditional_t<I == C, OutputT, InputT>;

    using DimsT = GemmDims;
    using ArgsT = GemmArgs<ComputeT>;
    static constexpr std::string_view group = "Gemm";
    static constexpr size_t buffer_count = 3;
    static constexpr std::array<bool, 3> is_output = {false, false, true};
    static constexpr std::array<Random::FillPolicy, 3> fill_policies = {
        Random::FillPolicy::Random, Random::FillPolicy::Random, Random::FillPolicy::Zero};

    static DimsT scale(size_t work_size) {
      double s = std::pow(static_cast<double>(work_size), 1.0 / 3.0);
      auto snap64 = [](double val) {
        size_t v = static_cast<size_t>(val);
        return std::max<size_t>(64, (v / 64) * 64);
      };
      return {snap64(512 * s), snap64(512 * s), snap64(512 * s)};
    };

    static std::array<size_t, 3> buffer_sizes(const GemmDims &d) {
      return {d.m * d.k, d.k * d.n, d.m * d.n};
    }
    static size_t flop_count(const GemmDims &d) {
      return 2 * d.m * d.n * d.k;
    }
    static size_t byte_count(const GemmDims &d) {
      return (d.m * d.k + d.k * d.n) * sizeof(InputT) // A + B
             + (d.m * d.n) * sizeof(OutputT);         // C
    }
    template <typename WorkloadT>
    static void register_options(WorkloadT &w, GemmDims &d) {
      w.add_option(group, "m", "Rows of A and C", d.m);
      w.add_option(group, "k", "Cols of A / rows of B", d.k);
      w.add_option(group, "n", "Cols of B and C", d.n);
    }
  };

} // namespace GpuBlas::Shapes
#endif // BLAS_BASELINER_BLASSHAPES_HPP