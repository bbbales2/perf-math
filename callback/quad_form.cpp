#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

static void quad_form_matrix_vector(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;
  
  auto init = [](benchmark::State& state) {
    Eigen::MatrixXd x_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));
    Eigen::VectorXd y_val = Eigen::VectorXd::Random(state.range(0));

    return std::make_tuple(promote_scalar<var>(x_val), promote_scalar<var>(y_val));
  };

  auto run = [](const auto&... args) {
    return sum(quad_form(args...));
  };

  callback_bench_impl(init, run, state);
}

static void quad_form_matrix_matrix(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;
  
  auto init = [](benchmark::State& state) {
    Eigen::MatrixXd x_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));
    Eigen::MatrixXd y_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));

    return std::make_tuple(promote_scalar<var>(x_val), promote_scalar<var>(y_val));
  };

  auto run = [](const auto&... args) {
    return sum(quad_form(args...));
  };

  callback_bench_impl(init, run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(quad_form_matrix_vector)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK(quad_form_matrix_matrix)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
