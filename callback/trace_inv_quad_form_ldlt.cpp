#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

static void trace_inv_quad_form_ldlt(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;
  
  auto init = [](benchmark::State& state) {
    Eigen::MatrixXd x_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));
    Eigen::MatrixXd y_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));

    stan::math::LDLT_factor<var, Eigen::Dynamic, Eigen::Dynamic>
    x_ldlt(promote_scalar<var>(x_val));
    
    return std::make_tuple(x_ldlt, promote_scalar<var>(y_val));
  };

  auto run = [](const auto&... args) {
    return trace_inv_quad_form_ldlt(args...);
  };

  callback_bench_impl(init, run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(trace_inv_quad_form_ldlt)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
