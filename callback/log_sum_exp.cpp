#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

static void log_sum_exp_vec(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;

  auto init = [](benchmark::State& state) {
    Eigen::VectorXd x_val = Eigen::VectorXd::Random(state.range(0));

    return std::make_tuple(promote_scalar<var>(x_val));
  };

  auto run = [](const auto&... args) {
    return sum(log_sum_exp(args...));
  };

  callback_bench_impl(init, run, state);
}

static void log_sum_exp_scalar(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;

  auto init = [](benchmark::State& state) {
    Eigen::VectorXd x_val = Eigen::VectorXd::Random(state.range(0));
    Eigen::VectorXd y_val = Eigen::VectorXd::Random(state.range(0));

    return std::make_tuple(promote_scalar<var>(x_val),
			   promote_scalar<var>(y_val));
  };

  auto run = [](const auto& arg1, const auto& arg2) {
    var out = 0.0;
    for(size_t i = 0; i < arg1.size(); ++i)
      out += log_sum_exp(arg1(i), arg2(i));
    return out;
  };

  callback_bench_impl(init, run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(log_sum_exp_vec)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(log_sum_exp_scalar)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();

