#ifndef BLAS_BASELINER_TYPES_HPP
#define BLAS_BASELINER_TYPES_HPP
#include <baseliner/core/Conversions.hpp>
#include <complex>
#include <string>
namespace GpuBlas::Types {
  struct DefaultMath {};
  template <typename T>
  struct ScalarInit {
    static T one() {
      return T{1.0};
    }
    static T zero() {
      return T{0.0};
    }
  };

  template <typename TypeT>
  struct ReferenceType {
    using type = TypeT;
    static constexpr double tolerance = 1e-6;
  };

  template <typename TypeT>
  struct ReferenceType<std::complex<TypeT>> {
    using type = std::complex<TypeT>;
    static constexpr double tolerance = 1e-6;
  };
  template <>
  struct ReferenceType<int32_t> {
    using type = int64_t;
    static constexpr double tolerance = 0.0;
  };

  template <typename TypeT>
  auto to_reference_type(const TypeT &val) -> typename ReferenceType<TypeT>::type {
    return static_cast<typename ReferenceType<TypeT>::type>(val);
  }

  template <typename TypeT>
  auto from_reference_type(const typename ReferenceType<TypeT>::type &val) -> TypeT {
    return static_cast<TypeT>(val);
  }

  template <typename InputTemplate, typename ComputeTemplate = InputTemplate, typename OutputTemplate = InputTemplate,
            typename PolicyTemplate = DefaultMath>
  struct TypeConfig {
    using InputT = InputTemplate;
    using ComputeT = ComputeTemplate;
    using OutputT = OutputTemplate;
    using ComputePolicyT = PolicyTemplate;
    using ReferenceComputeT = typename ReferenceType<ComputeT>::type;
  };
  // Type letter traits
  template <typename T>
  struct TypeLetter;
  template <>
  struct TypeLetter<float> {
    static constexpr const char *value = "s";
  };
  template <>
  struct TypeLetter<int> {
    static constexpr const char *value = "i32";
  };
  template <>
  struct TypeLetter<double> {
    static constexpr const char *value = "d";
  };
  template <>
  struct TypeLetter<int8_t> {
    static constexpr const char *value = "i8";
  };
  template <>
  struct TypeLetter<int64_t> {
    static constexpr const char *value = "i64";
  };

  // Policy name traits
  template <typename T>
  struct PolicyName {
    static constexpr const char *value = "";
  }; // default = silent
  template <>
  struct PolicyName<DefaultMath> {
    static constexpr const char *value = "";
  };

  template <typename TypeConfig, typename DimType>
  auto type_config_name() -> std::string {
    using In = typename TypeConfig::InputT;
    using Compute = typename TypeConfig::ComputeT;
    using Out = typename TypeConfig::OutputT;
    using Policy = typename TypeConfig::ComputePolicyT;

    constexpr bool uniform = std::is_same_v<In, Compute> && std::is_same_v<Compute, Out>;
    constexpr bool isint = std::is_same_v<int, DimType>;

    std::string name;
    if constexpr (uniform) {
      name = TypeLetter<In>::value;
    } else {
      name = std::string(TypeLetter<In>::value) + TypeLetter<Compute>::value + TypeLetter<Out>::value;
    }
    if (!isint) {
      name += TypeLetter<DimType>::value;
    }
    name += PolicyName<Policy>::value;
    return name;
  };

  template <typename TypeT>
  auto to_string(TypeT val) -> std::string {
    return Baseliner::Conversion::baseliner_to_string(to_reference_type<TypeT>(val));
  };

  template <typename TypeT>
  auto from_string(std::string val) -> TypeT {
    using RefT = typename ReferenceType<TypeT>::type;
    RefT ref_val = Baseliner::Conversion::baseliner_from_string<RefT>(val);
    return from_reference_type<TypeT>(ref_val);
  }

  template <typename T>
  struct TypeOperations {
    static constexpr size_t factor = 1;
  };

} // namespace GpuBlas::Types
namespace Baseliner::Conversion {

  template <typename T>
  inline auto complex_to_string(const std::complex<T> &val) -> std::string {
    return "{" + baseliner_to_string(val.real()) + " ," + baseliner_to_string(val.imag()) + "}";
  }
  template <typename T>
  auto complex_from_string(const std::string &val) -> std::complex<T> {
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

    return {baseliner_from_string<T>(first_part), baseliner_from_string<T>(second_part)};
  }

  template <>
  inline auto baseliner_to_string<std::complex<float>>(const std::complex<float> &val) -> std::string {
    return complex_to_string(val);
  };
  template <>
  inline auto baseliner_from_string(const std::string &val) -> std::complex<float> {
    return complex_from_string<float>(val);
  };
  template <>
  inline auto baseliner_to_string<std::complex<double>>(const std::complex<double> &val) -> std::string {
    return complex_to_string(val);
  };
  template <>
  inline auto baseliner_from_string(const std::string &val) -> std::complex<double> {
    return complex_from_string<double>(val);
  };
} // namespace Baseliner::Conversion
#endif // BLAS_BASELINER_TYPES_HPP