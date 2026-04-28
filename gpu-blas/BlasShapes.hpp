#ifndef BLAS_BASELINER_BLASSHAPES_HPP
#define BLAS_BASELINER_BLASSHAPES_HPP
#include <baseliner/core/Options.hpp>
#include <functional>
#include <gpu-blas/Buffers.hpp>
#include <gpu-blas/Random.hpp>
#include <gpu-blas/Types.hpp>
#include <gpu-blas/Validation.hpp>

namespace GpuBlas {
  namespace Shapes {

    // StandardGemm
    template <typename T>
    struct GemmDims : public Baseliner::IOption {
      T m = 512;
      T n = 512;
      T k = 512;

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

      static void scale(DimsT &dims, size_t work_size) {
        double s = std::pow(static_cast<double>(work_size), 1.0 / 3.0);
        auto snap64 = [](double val) -> size_t {
          size_t v = static_cast<size_t>(val);
          return std::max<size_t>(64, (v / 64) * 64);
        };
        dims = DimsT(snap64(dims.m * s), snap64(dims.n * s), snap64(dims.k * s));
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
      static bool validate(const Buffers<GemmShape> &buffers, const DimsT &dims, const ArgsT &args, float &mean_error,
                           size_t samples = 1) {

        using Shape = GemmShape<TypeConfigT, DimTypes>;

        bool ok = true;

        ok &= GpuBlas::Validation::gemm_spot_check<Shape>(buffers, dims, args, mean_error, samples);

        return ok;
      }
    };

  } // namespace Shapes
} // namespace GpuBlas
#endif // BLAS_BASELINER_BLASSHAPES_HPP