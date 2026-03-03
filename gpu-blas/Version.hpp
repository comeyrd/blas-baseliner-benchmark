#pragma once
#include <string>
#include <string_view>
namespace GpuBlas::Version {
inline constexpr int major = 0; // NOLINT
inline constexpr int minor = 0; // NOLINT
inline constexpr int patch = 1; // NOLINT
inline constexpr std::string_view string_view = "0.0.1";
inline auto string() -> std::string { return std::string(string_view); }
} // namespace GpuBlas::Version