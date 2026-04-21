#pragma once
#include <baseliner/core/stats/IStats.hpp>

class MeanError : public Baseliner::Stats::Imetric<MeanError, float> {
public:
  using Imetric<MeanError, float>::Imetric; // Needs this for defaults
  [[nodiscard]] auto name() const -> std::string override {
    return "mean_error";
  }
  [[nodiscard]] auto unit() const -> std::string override {
    return "";
  }
  [[nodiscard]] auto saving_policy() -> Baseliner::Stats::MetricSavingPolicy override {
    return Baseliner::Stats::MetricSavingPolicy::SAVE;
  }
};