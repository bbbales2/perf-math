CXXFLAGS = -march=native -mtune=native

-include Makefile

BENCHMARKS = sd squared_distance variance sum unit_vector_constrain quad_form trace_quad_form trace_gen_quad_form multiply mdivide_left mdivide_left_spd mdivide_left_tri matrix_power softmax log_softmax log_determinant_spd log_determinant log_sum_exp dot_product divide gp_exp_quad_cov gp_periodic_cov dot_self columns_dot_self tcrossprod determinant inverse multiply_lower_tri_self_transpose columns_dot_product rows_dot_product cauchy chi_square double_exponential exp_mod_normal exponential frechet ordered_constrain positive_ordered_constrain simplex_constrain
BENCHMARK_BINARIES = $(patsubst %,callback/%,$(BENCHMARKS))
BENCHMARK_CSVS = $(patsubst %,callback/%.csv,$(BENCHMARKS))

.PHONY : clean_callback build

$(BENCHMARK_CSVS) : callback/%.csv : callback/%
	$< --benchmark_color=false --benchmark_repetitions=30 --benchmark_min_time=0.1 --benchmark_format=csv | tail -n +2 > $@

callback/benchmark.csv : $(BENCHMARK_CSVS)
	echo "name,iterations,real_time,cpu_time,time_unit,bytes_per_second,items_per_second,label,error_occurred,error_message" > $@
	cat $^ >> $@

build : $(BENCHMARK_BINARIES)

clean_callback :
	-@rm callback/benchmark.csv
	-@rm $(BENCHMARK_CSVS)
	-@rm $(BENCHMARK_BINARIES)
