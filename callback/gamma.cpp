#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

auto init = [](benchmark::State& state) {
  using stan::math::var;
  using stan::math::exp;
  using stan::math::promote_scalar;

  Eigen::VectorXd y_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd alpha_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd beta_val = exp(Eigen::VectorXd::Random(state.range(0)));

  return std::make_tuple(promote_scalar<var>(y_val),
			 promote_scalar<var>(alpha_val),
			 promote_scalar<var>(beta_val));
};

auto init_data = [](benchmark::State& state) {
  using stan::math::var;
  using stan::math::exp;
  using stan::math::promote_scalar;

  Eigen::VectorXd y_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd alpha_val = exp(Eigen::VectorXd::Random(state.range(0)));
  Eigen::VectorXd beta_val = exp(Eigen::VectorXd::Random(state.range(0)));

  return std::make_tuple(y_val,
			 promote_scalar<var>(alpha_val),
			 promote_scalar<var>(beta_val));
};

static void gamma_lpdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_lpdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void gamma_cdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_cdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void gamma_lcdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_lcdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void gamma_lccdf(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_lccdf(args...);
  };

  callback_bench_impl(init, run, state);
}

static void gamma_lpdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_lpdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

static void gamma_cdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_cdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

static void gamma_lcdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_lcdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

static void gamma_lccdf_data(benchmark::State& state) {
  auto run = [](const auto&... args) {
    return gamma_lccdf(args...);
  };

  callback_bench_impl(init_data, run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(gamma_lpdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gamma_cdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gamma_lcdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gamma_lccdf)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gamma_lpdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gamma_cdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gamma_lcdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(gamma_lccdf_data)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
