#include "../GpuBlas.hpp"
#include "../Random.hpp"
#include "cublas_v2.h"
#include <baseliner/core/AxeSweeping.hpp>
#include <baseliner/core/Conversions.hpp>

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
    template <>
    struct RandomTraits<__half> {
      using ScalarT = float; // Use float for the random math
      template <typename Op>
      static __half create(Op &&gen) {
        float val = gen();
        return __float2half(val);
      }
    };

  } // namespace Random
} // namespace GpuBlas
namespace Detail {
  template <typename ComplexT, typename ScalarT>
  struct ComplexSweeperImpl {
    static auto generate(const ::Baseliner::TypedSweepHint<ComplexT> &hint) -> std::vector<ComplexT> {
      if (hint.m_policy == ::Baseliner::SweepPolicy::Enumerated) {
        return hint.m_enumerated;
      }

      std::vector<ComplexT> result;

      if (hint.m_policy == ::Baseliner::SweepPolicy::LinearRange) {
        if (hint.m_step.x <= static_cast<ScalarT>(0) || hint.m_step.y <= static_cast<ScalarT>(0)) {
          throw std::invalid_argument("Step components must be > 0");
        }
        for (ScalarT x = hint.m_min.x; x <= hint.m_max.x; x += hint.m_step.x) {
          for (ScalarT y = hint.m_min.y; y <= hint.m_max.y; y += hint.m_step.y) {
            result.push_back({x, y});
          }
        }
      } else if (hint.m_policy == ::Baseliner::SweepPolicy::PowersOfTwo) {
        if (hint.m_min.x <= static_cast<ScalarT>(0) || hint.m_min.y <= static_cast<ScalarT>(0)) {
          throw std::invalid_argument("Min components must be > 0 for PowersOfTwo");
        }
        for (ScalarT x = hint.m_min.x; x <= hint.m_max.x; x *= static_cast<ScalarT>(2)) {
          for (ScalarT y = hint.m_min.y; y <= hint.m_max.y; y *= static_cast<ScalarT>(2)) {
            result.push_back({x, y});
          }
        }
      }

      return result;
    }
  };
} // namespace Detail

namespace Baseliner::Sweep::Detail {
  template <>
  struct Sweeper<cuComplex> : ::Detail::ComplexSweeperImpl<cuComplex, float> {};

  template <>
  struct Sweeper<cuDoubleComplex> : ::Detail::ComplexSweeperImpl<cuDoubleComplex, double> {};

} // namespace Baseliner::Sweep::Detail

namespace Baseliner::Conversion {

  template <typename XY>
  inline auto xy_to_string(const XY &val) -> std::string {
    return "{" + baseliner_to_string(val.x) + " ," + baseliner_to_string(val.y) + "}";
  }
  template <typename XY, typename F>
  auto xy_from_string(const std::string &val) -> XY {
    std::string string_v = trim_before_after_whitespace(val);

    if (string_v.size() >= 2 && ((string_v.front() == '{' && string_v.back() == '}'))) {
      string_v = string_v.substr(1, string_v.size() - 2);
    }
    size_t comma_pos = string_v.find(',');
    if (comma_pos == std::string::npos) {
      throw std::invalid_argument("Input does not contain a pair separator: " + val);
    }
    std::string first_part = trim_before_after_whitespace(string_v.substr(0, comma_pos));
    std::string second_part = trim_before_after_whitespace(string_v.substr(comma_pos + 1));

    return {baseliner_from_string<F>(first_part), baseliner_from_string<F>(second_part)};
  }
  template <>
  inline auto baseliner_to_string<cuComplex>(const cuComplex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<cuComplex>(const std::string &val) -> cuComplex {
    return xy_from_string<cuComplex, float>(val);
  };
  template <>
  inline auto baseliner_to_string<cuDoubleComplex>(const cuDoubleComplex &val) -> std::string {
    return xy_to_string(val);
  };
  template <>
  inline auto baseliner_from_string<cuDoubleComplex>(const std::string &val) -> cuDoubleComplex {
    return xy_from_string<cuDoubleComplex, double>(val);
  };
  template <>
  inline auto baseliner_from_string<__half>(const std::string &val) -> __half {
    float f_val = std::stof(val);
    return __half{__float2half(f_val)};
  }

  template <>
  inline auto baseliner_to_string<__half>(const __half &val) -> std::string {
    float f_val;
    f_val = __half2float(val);
    return std::to_string(f_val);
  }
} // namespace Baseliner::Conversion
