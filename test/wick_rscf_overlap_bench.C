#include "wick_bench_common.h"

int main()
{
    return run_rscf_benchmark(
        "h2o_6-31g",
        "wick_rscf_overlap_bench",
        wick_bench_task::overlap);
}
