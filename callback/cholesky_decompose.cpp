#include <benchmark/benchmark.h>
#include <stan/math.hpp>
#include <utility>
#include "toss_me.hpp"
#include "callback_bench_impl.hpp"

static void cholesky_decompose(benchmark::State& state) {
  using stan::math::var;
  using stan::math::promote_scalar;

  auto init = [](benchmark::State& state) {
    Eigen::MatrixXd x_val = Eigen::MatrixXd::Random(state.range(0), state.range(0));

    x_val = (x_val + x_val.transpose()).eval();
    for(size_t i = 0; i < x_val.rows(); ++i)
      x_val(i, i) += 2.0 * x_val.cols();

    return std::make_tuple(promote_scalar<var>(x_val));
  };

  auto run = [](const auto&... args) {
    return sum(cholesky_decompose(args...));
  };

  callback_bench_impl(init, run, state);
}

// The start and ending sizes for the benchmark
int start_val = 2;
int end_val = 1024;
BENCHMARK(toss_me);
BENCHMARK(cholesky_decompose)->RangeMultiplier(2)->Range(start_val, end_val)->UseManualTime();
BENCHMARK_MAIN();
