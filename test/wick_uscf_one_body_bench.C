#include "wick_bench_common.h"

int main()
{
    return run_uscf_benchmark(
        "h2o_6-31g",
        "wick_uscf_one_body_bench",
        wick_bench_task::one_body);
}
