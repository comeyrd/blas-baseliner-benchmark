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
      T m = 256;
      T n = 256;
      T k = 256;
      size_t B = 64;
      GemmDims() = default;
      GemmDims(T m_, T n_, T k_, size_t B_)
          : m(m_),
            n(n_),
            k(k_),
            B(B_) {};
      GemmDims &operator=(const GemmDims &other) {
        if (this != &other) {
          m = other.m;
          n = other.n;
          k = other.k;
          B = other.B;
        }
        return *this;
      }
      void register_options() override {
        this->add_option("GemmDims", "m", "Rows of A and C", m);
        this->add_option("GemmDims", "k", "Cols of A / rows of B", k);
        this->add_option("GemmDims", "n", "Cols of B and C", n);
        this->add_option("GemmDims", "alignment_block", "To a multiple of what the dimensions should snap to?", B);
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
      using DimTypesT = DimTypes;
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

      static void scale(DimsT &dims, size_t work_size, const size_t &batch_count) {
        size_t safe_batch = std::max<size_t>(1, batch_count);

        double target_ratio = static_cast<double>(work_size) / (safe_batch * Types::TypeOperations<InputT>::factor);
        double s = std::pow(target_ratio, 1.0 / 3.0);

        auto snap = [&](double val) -> size_t {
          size_t b = std::max<size_t>(1, dims.B);
          size_t v_aligned = static_cast<size_t>(std::ceil(val / b)) * b;
          return std::max<size_t>(b, v_aligned);
        };

        dims.m = snap(dims.m * s);
        dims.n = snap(dims.n * s);
        dims.k = snap(dims.k * s);
      }

      static std::array<size_t, 2> input_buffer_sizes(const DimsT &d) {
        return {{static_cast<size_t>(d.m) * d.k, static_cast<size_t>(d.k) * d.n}};
      }

      static std::array<size_t, 1> output_buffer_sizes(const DimsT &d) {
        return {{static_cast<size_t>(d.m) * d.n}};
      }

      static size_t flop_count(const DimsT &d, const size_t &batch_size) {
        return 2ULL * d.m * d.n * d.k * batch_size * Types::TypeOperations<InputT>::factor;
      }
      static size_t byte_count(const DimsT &d, const size_t &batch_size) {
        return (d.m * d.k + d.k * d.n * batch_size) * sizeof(InputT) // A + B
               + (d.m * d.n * batch_size) * sizeof(OutputT);         // C
      }
      static bool validate(const Buffers<GemmShape> &buffers, const DimsT &dims, const ArgsT &args, float &mean_error,
                           const size_t &batch_size, size_t samples = 1) {

        using Shape = GemmShape<TypeConfigT, DimTypes>;

        bool ok = true;
        std::cout << "Validating\n";
        ok &= GpuBlas::Validation::gemm_spot_check<Shape>(buffers, dims, args, mean_error, batch_size, samples);

        return ok;
      }
    };

  } // namespace Shapes
} // namespace GpuBlas
#endif // BLAS_BASELINER_BLASSHAPES_HPP