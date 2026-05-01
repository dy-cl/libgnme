#include "wick_bench_common.h"

int main()
{
    return run_uscf_benchmark(
        "h2o_6-31g",
        "wick_uscf_overlap_bench",
        wick_bench_task::overlap);
}
