#pragma once
#include <baseliner/core/stats/IStats.hpp>
#include <baseliner/core/stats/StatsType.hpp>

class MeanError : public Baseliner::Stats::Imetric<MeanError, float> {
public:
  using Imetric<MeanError, float>::Imetric; // Needs this for defaults
  [[nodiscard]] auto name() const -> std::string override {
    return "mean_error";
  }
  [[nodiscard]] auto unit() const -> std::string override {
    return "";
  }
  [[nodiscard]] auto saving_policy() const -> Baseliner::Stats::SavingPolicy override {
    return Baseliner::Stats::SavingPolicy::SAVE;
  }
  [[nodiscard]] auto granularity() const -> Baseliner::MetricGranularity override {
    return Baseliner::MetricGranularity::EVERY_BATCH;
  };
};