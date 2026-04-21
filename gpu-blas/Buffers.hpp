#pragma once
#include <array>
#include <cstddef>
#include <vector>
template <typename ShapeT>
struct Buffers {
  static constexpr size_t I_ = ShapeT::input_counts;
  static constexpr size_t O_ = ShapeT::output_counts;

  std::array<typename ShapeT::InputT *, I_> input_device{};
  std::array<std::vector<typename ShapeT::InputT>, I_> input_host;
  std::array<typename ShapeT::OutputT *, O_> output_device{};
  std::array<std::vector<typename ShapeT::OutputT>, O_> output_host;

  auto in_device(size_t i) -> typename ShapeT::InputT * {
    return static_cast<typename ShapeT::InputT *>(input_device[i]);
  }
  auto out_device(size_t i) -> typename ShapeT::OutputT * {
    return static_cast<typename ShapeT::OutputT *>(output_device[i]);
  }
};