#ifndef LIBGNME_TEST_WICK_BENCH_COMMON_H
#define LIBGNME_TEST_WICK_BENCH_COMMON_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <armadillo>

#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/wick/wick_rscf.h>
#include <libgnme/wick/wick_uscf.h>
#include "testing.h"

using namespace libgnme;

/** \brief Timer that reports elapsed wall time on destruction.
 **/
struct timer
{
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    double *elapsed;

    timer(const std::string &_name, double *_elapsed = 0) :
        name(_name),
        start(std::chrono::high_resolution_clock::now()),
        elapsed(_elapsed)
    { }

    ~timer()
    {
        const auto stop = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(stop - start).count();

        if(elapsed) *elapsed = dt;

        std::cout << name << ": "
                  << std::fixed << std::setprecision(6)
                  << dt << " s" << std::endl;
    }
};

/** \brief Type of Wick matrix element benchmark.
 **/
enum class wick_bench_task
{
    overlap,
    one_body,
    two_body
};

/** \brief Spin-resolved excitation class.
 **/
struct excitation_class
{
    std::string name;
    size_t arank;
    size_t brank;
};

/** \brief One benchmark timing record.
 **/
struct benchmark_record
{
    std::string name;
    size_t cases;
    double seconds;
    double checksum_delta;
};

/** \brief Recursively generate combinations.
 **/
inline void combinations_rec(
    const size_t start,
    const size_t stop,
    const size_t rank,
    std::vector<size_t> &cur,
    std::vector<std::vector<size_t> > &out)
{
    if(cur.size() == rank)
    {
        out.push_back(cur);
        return;
    }

    for(size_t i=start; i<stop; i++)
    {
        cur.push_back(i);
        combinations_rec(i+1, stop, rank, cur, out);
        cur.pop_back();
    }
}

/** \brief Generate all index combinations of a given rank.
 **/
inline std::vector<std::vector<size_t> > combinations(
    const size_t stop,
    const size_t rank)
{
    std::vector<size_t> cur;
    std::vector<std::vector<size_t> > out;

    combinations_rec(0, stop, rank, cur, out);

    return out;
}

/** \brief Build particle-hole excitation index arrays.
 **/
inline std::vector<arma::umat> excitations(
    const size_t nocc,
    const size_t nmo,
    const size_t rank,
    const size_t max_virt,
    const size_t max_out)
{
    std::vector<arma::umat> out;

    if(rank == 0)
    {
        out.push_back(arma::umat(0,2));
        return out;
    }

    if(rank > nocc) return out;

    const size_t nvirt = std::min(max_virt, nmo - nocc);
    if(rank > nvirt) return out;

    std::vector<std::vector<size_t> > holes = combinations(nocc, rank);
    std::vector<std::vector<size_t> > parts = combinations(nvirt, rank);

    for(size_t ih=0; ih<holes.size(); ih++)
    for(size_t ip=0; ip<parts.size(); ip++)
    {
        arma::umat hp(rank,2);

        for(size_t k=0; k<rank; k++)
        {
            hp(k,0) = holes[ih][k];
            hp(k,1) = nocc + parts[ip][k];
        }

        out.push_back(hp);

        if(out.size() >= max_out)
            return out;
    }

    return out;
}

/** \brief Human-readable benchmark task name.
 **/
inline std::string task_name(const wick_bench_task task)
{
    if(task == wick_bench_task::overlap)  return "overlap";
    if(task == wick_bench_task::one_body) return "one-body";
    return "two-body";
}

/** \brief Human-readable rank name.
 **/
inline std::string rank_name(const size_t rank)
{
    if(rank == 0) return "ref";
    if(rank == 1) return "single";
    if(rank == 2) return "double";
    if(rank == 3) return "triple";
    if(rank == 4) return "quad";

    std::ostringstream oss;
    oss << "rank" << rank;
    return oss.str();
}

/** \brief Human-readable spin-resolved excitation class name.
 **/
inline std::string class_name(const size_t arank, const size_t brank)
{
    if(arank == 0 && brank == 0)
        return "ref";

    if(arank != 0 && brank == 0)
        return "alpha " + rank_name(arank);

    if(arank == 0 && brank != 0)
        return "beta " + rank_name(brank);

    std::ostringstream oss;
    oss << "mixed " << rank_name(arank + brank)
        << " a" << arank << "b" << brank;

    return oss.str();
}

/** \brief Build all spin-resolved excitation classes up to total rank four.
 **/
inline std::vector<excitation_class> excitation_classes()
{
    std::vector<excitation_class> out;

    for(size_t total=0; total<=4; total++)
    for(size_t arank=0; arank<=total; arank++)
    {
        const size_t brank = total - arank;

        excitation_class c;
        c.name  = class_name(arank, brank);
        c.arank = arank;
        c.brank = brank;

        out.push_back(c);
    }

    return out;
}

/** \brief Choose a case limit for a pair of excitation classes.
 **/
inline size_t max_cases_for(
    const excitation_class &bra,
    const excitation_class &ket)
{
    const size_t total = bra.arank + bra.brank + ket.arank + ket.brank;

    if(total <= 2) return 8192;
    if(total <= 4) return 4096;
    if(total <= 6) return 2048;

    return 1024;
}

/** \brief Count tensor-product cases with a maximum cap.
 **/
inline size_t count_cases(
    const std::vector<arma::umat> &xa,
    const std::vector<arma::umat> &xb,
    const std::vector<arma::umat> &wa,
    const std::vector<arma::umat> &wb,
    const size_t max_cases)
{
    size_t ncase = 0;

    for(size_t i=0; i<xa.size(); i++)
    for(size_t j=0; j<xb.size(); j++)
    for(size_t k=0; k<wa.size(); k++)
    for(size_t l=0; l<wb.size(); l++)
    {
        ncase++;

        if(ncase >= max_cases)
            return ncase;
    }

    return ncase;
}

/** \brief Evaluate one Wick matrix element for a USCF builder.
 **/
inline void evaluate_case(
    wick_uscf<double,double,double> &mb,
    const wick_bench_task task,
    arma::umat &xa,
    arma::umat &xb,
    arma::umat &wa,
    arma::umat &wb,
    double &swick,
    double &vwick)
{
    if(task == wick_bench_task::overlap)
    {
        mb.evaluate_overlap(xa, xb, wa, wb, swick);
        vwick = 0.0;
        return;
    }

    mb.evaluate(xa, xb, wa, wb, swick, vwick);
}

/** \brief Evaluate one Wick matrix element for a RSCF builder.
 **/
inline void evaluate_case(
    wick_rscf<double,double,double> &mb,
    const wick_bench_task task,
    arma::umat &xa,
    arma::umat &xb,
    arma::umat &wa,
    arma::umat &wb,
    double &swick,
    double &vwick)
{
    if(task == wick_bench_task::overlap)
    {
        mb.evaluate_overlap(xa, xb, wa, wb, swick);
        vwick = 0.0;
        return;
    }

    mb.evaluate(xa, xb, wa, wb, swick, vwick);
}

/** \brief Run a collection of Wick matrix element evaluations.
 **/
template<typename Wick>
inline void run_cases(
    const std::string &name,
    Wick &mb,
    const wick_bench_task task,
    const std::vector<arma::umat> &xa,
    const std::vector<arma::umat> &xb,
    const std::vector<arma::umat> &wa,
    const std::vector<arma::umat> &wb,
    const size_t max_cases,
    double &checksum,
    std::vector<benchmark_record> &records)
{
    const size_t ncase = count_cases(xa, xb, wa, wb, max_cases);

    if(ncase == 0)
        return;

    double swick = 0.0;
    double vwick = 0.0;
    double elapsed = 0.0;
    const double checksum_before = checksum;

    std::cout << name << " cases: " << ncase << std::endl;

    size_t count = 0;

    {
        timer t(name, &elapsed);

        for(size_t i=0; i<xa.size() && count<max_cases; i++)
        for(size_t j=0; j<xb.size() && count<max_cases; j++)
        for(size_t k=0; k<wa.size() && count<max_cases; k++)
        for(size_t l=0; l<wb.size() && count<max_cases; l++)
        {
            // Copy particle-hole arrays because evaluate takes mutable references
            arma::umat xac = xa[i];
            arma::umat xbc = xb[j];
            arma::umat wac = wa[k];
            arma::umat wbc = wb[l];

            // Evaluate Wick matrix elements
            evaluate_case(mb, task, xac, xbc, wac, wbc, swick, vwick);

            // Accumulate checksum to prevent dead-code elimination
            checksum += swick + vwick;

            count++;
        }
    }

    benchmark_record rec;
    rec.name = name;
    rec.cases = count;
    rec.seconds = elapsed;
    rec.checksum_delta = checksum - checksum_before;

    records.push_back(rec);
}

/** \brief Run one pair of spin-resolved excitation classes.
 **/
template<typename Wick>
inline void run_class_pair(
    Wick &mb,
    const wick_bench_task task,
    const excitation_class &bra,
    const excitation_class &ket,
    const std::vector<std::vector<arma::umat> > &ea,
    const std::vector<std::vector<arma::umat> > &eb,
    double &checksum,
    std::vector<benchmark_record> &records)
{
    std::ostringstream oss;
    oss << "mb.evaluate " << bra.name << "/" << ket.name;

    const size_t max_cases = max_cases_for(bra, ket);

    run_cases(
        oss.str(),
        mb,
        task,
        ea[bra.arank], eb[bra.brank],
        ea[ket.arank], eb[ket.brank],
        max_cases,
        checksum,
        records);
}

/** \brief Print benchmark summary table.
 **/
inline void print_benchmark_summary(
    const char *bench_name,
    const wick_bench_task task,
    const std::vector<benchmark_record> &records,
    const double checksum)
{
    double total = 0.0;
    size_t total_cases = 0;

    for(size_t i=0; i<records.size(); i++)
    {
        total += records[i].seconds;
        total_cases += records[i].cases;
    }

    std::vector<benchmark_record> sorted = records;

    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const benchmark_record &a, const benchmark_record &b) {
            return a.seconds > b.seconds;
        });

    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Benchmark summary: " << bench_name << std::endl;
    std::cout << "Task: " << task_name(task) << std::endl;
    std::cout << "Blocks: " << records.size() << std::endl;
    std::cout << "Cases: " << total_cases << std::endl;
    std::cout << "Timed case total: " << std::fixed << std::setprecision(6)
              << total << " s" << std::endl;

    if(total_cases != 0)
    {
        std::cout << "Mean time per case: "
                  << std::fixed << std::setprecision(6)
                  << 1.0e6 * total / static_cast<double>(total_cases)
                  << " us" << std::endl;
    }

    std::cout << "Checksum: " << std::setprecision(16)
              << checksum << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::cout << std::left
              << std::setw(48) << "Block"
              << std::right
              << std::setw(12) << "Cases"
              << std::setw(14) << "Time / s"
              << std::setw(14) << "us/case"
              << std::setw(10) << "%"
              << std::endl;

    std::cout << "------------------------------------------------------------" << std::endl;

    for(size_t i=0; i<sorted.size(); i++)
    {
        const double us_case =
            sorted[i].cases == 0 ? 0.0 :
            1.0e6 * sorted[i].seconds / static_cast<double>(sorted[i].cases);

        const double frac =
            total == 0.0 ? 0.0 : 100.0 * sorted[i].seconds / total;

        std::cout << std::left
                  << std::setw(48) << sorted[i].name
                  << std::right
                  << std::setw(12) << sorted[i].cases
                  << std::setw(14) << std::fixed << std::setprecision(6)
                  << sorted[i].seconds
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << us_case
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << frac
                  << std::endl;
    }

    std::cout << "============================================================" << std::endl;
}

/** \brief Read common benchmark input data.
 **/
inline int read_benchmark_data(
    const char *testcase,
    size_t &nbsf,
    size_t &nocca,
    size_t &noccb,
    size_t &nmo,
    size_t &nact,
    arma::mat &S,
    arma::mat &II,
    arma::mat &Cread)
{
    // Filenames
    std::string fname_nmo = testcase + std::string("/nelec.txt");
    std::string fname_c   = testcase + std::string("/coeff.txt");
    std::string fname_ov  = testcase + std::string("/ovls.txt");
    std::string fname_II  = testcase + std::string("/teis.bin");

    // Read-in basis information
    std::ifstream nelec_file;
    nelec_file.open(fname_nmo);
    if(!nelec_file)
    {
        std::cerr << "Could not open " << fname_nmo << std::endl;
        return 1;
    }
    nelec_file >> nbsf >> nocca >> noccb >> nact;
    nelec_file.close();

    nmo = nbsf;

    // Check the input
    assert(nmo   > 0);
    assert(nbsf  > 0);
    assert(nocca > 0);
    assert(noccb > 0);

    // Read-in overlap matrix
    if(!S.load(fname_ov, arma::raw_ascii))
    {
        std::cerr << "Could not open " << fname_ov << std::endl;
        return 1;
    }
    assert(S.n_rows == nbsf);
    assert(S.n_cols == nbsf);

    // Read-in two-electron integrals
    if(!II.load(fname_II, arma::raw_binary))
    {
        std::cerr << "Could not open " << fname_II << std::endl;
        return 1;
    }
    assert(II.n_elem == nbsf*nbsf*nbsf*nbsf);
    II.reshape(nbsf*nbsf, nbsf*nbsf);

    // Read-in orbital coefficients
    if(!Cread.load(fname_c, arma::raw_ascii))
    {
        std::cerr << "Could not open " << fname_c << std::endl;
        return 1;
    }
    assert(Cread.n_rows == nbsf);
    assert(Cread.n_cols == 2*nmo);

    return 0;
}

/** \brief Construct a deterministic AO one-body matrix for timing.
 **/
inline arma::mat one_body_matrix(const arma::mat &S)
{
    // Use the overlap matrix as a deterministic one-body operator
    arma::mat F = S;
    return F;
}

/** \brief Run the systematic USCF benchmark.
 **/
inline int run_uscf_benchmark(
    const char *testcase,
    const char *bench_name,
    const wick_bench_task task)
{
    // Report who we are
    std::cout << bench_name << "::systematic_spin_rank_benchmark("
              << testcase << ")" << std::endl;

    // Read input data
    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0;
    arma::mat S, II, Cread;

    if(read_benchmark_data(testcase, nbsf, nocca, noccb, nmo, nact, S, II, Cread))
        return 1;

    // Initialise memory for bra and ket orbitals with memory access views
    arma::mat Cx(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cx_a(Cx.colptr(0), nbsf, nmo, false, true);
    arma::mat Cx_b(Cx.colptr(nmo), nbsf, nmo, false, true);

    arma::mat Cw(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cw_a(Cw.colptr(0), nbsf, nmo, false, true);
    arma::mat Cw_b(Cw.colptr(nmo), nbsf, nmo, false, true);

    // Define bra and ket orbital coefficients as spin-flip pairs
    Cx_a = Cread.cols(0,nmo-1);
    Cw_b = Cread.cols(0,nmo-1);
    Cx_b = Cread.cols(nmo,2*nmo-1);
    Cw_a = Cread.cols(nmo,2*nmo-1);

    // Construct the wick_orbitals objects
    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    wick_uscf<double,double,double> mb(orbs_a, orbs_b);

    if(task == wick_bench_task::one_body)
    {
        arma::mat F = one_body_matrix(S);

        timer t("mb.add_one_body");
        mb.add_one_body(F);
    }

    if(task == wick_bench_task::two_body)
    {
        timer t("mb.add_two_body");
        mb.add_two_body(II);
    }

    // Define excitation-generation limits
    const size_t max_virt = 6;
    const size_t max_exc  = 500;

    // Generate alpha and beta excitations up to quadruples
    std::vector<std::vector<arma::umat> > ea(5);
    std::vector<std::vector<arma::umat> > eb(5);

    for(size_t rank=0; rank<=4; rank++)
    {
        ea[rank] = excitations(nocca, nmo, rank, max_virt, max_exc);
        eb[rank] = excitations(noccb, nmo, rank, max_virt, max_exc);

        std::cout << "alpha " << rank_name(rank) << " excitations: "
                  << ea[rank].size() << std::endl;
        std::cout << "beta  " << rank_name(rank) << " excitations: "
                  << eb[rank].size() << std::endl;
    }

    // Build spin-resolved excitation classes
    std::vector<excitation_class> classes = excitation_classes();

    // Evaluate every ordered pair of classes up to mixed quad/mixed quad
    double checksum = 0.0;
    std::vector<benchmark_record> records;

    for(size_t i=0; i<classes.size(); i++)
    for(size_t j=0; j<classes.size(); j++)
        run_class_pair(mb, task, classes[i], classes[j], ea, eb, checksum, records);

    // Report checksum and timing summary
    print_benchmark_summary(bench_name, task, records, checksum);

    if(!std::isfinite(checksum)) return 1;

    return 0;
}

/** \brief Run the systematic RSCF benchmark.
 **/
inline int run_rscf_benchmark(
    const char *testcase,
    const char *bench_name,
    const wick_bench_task task)
{
    // Report who we are
    std::cout << bench_name << "::systematic_spin_rank_benchmark("
              << testcase << ")" << std::endl;

    // Read input data
    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0;
    arma::mat S, II, Cread;

    if(read_benchmark_data(testcase, nbsf, nocca, noccb, nmo, nact, S, II, Cread))
        return 1;

    // The restricted benchmark assumes a closed-shell reference
    assert(nocca == noccb);

    // Initialise memory for bra and ket orbitals
    arma::mat Cx(nbsf, nmo, arma::fill::zeros);
    arma::mat Cw(nbsf, nmo, arma::fill::zeros);

    // Define bra and ket orbital coefficients as a nonorthogonal pair
    Cx = Cread.cols(0,nmo-1);
    Cw = Cread.cols(nmo,2*nmo-1);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs(nbsf, nmo, nocca, Cx, Cw, S);

    // Setup matrix builder
    wick_rscf<double,double,double> mb(orbs);

    if(task == wick_bench_task::one_body)
    {
        arma::mat F = one_body_matrix(S);

        timer t("mb.add_one_body");
        mb.add_one_body(F);
    }

    if(task == wick_bench_task::two_body)
    {
        timer t("mb.add_two_body");
        mb.add_two_body(II);
    }

    // Define excitation-generation limits
    const size_t max_virt = 6;
    const size_t max_exc  = 500;

    // Generate alpha and beta excitations up to quadruples
    std::vector<std::vector<arma::umat> > ea(5);
    std::vector<std::vector<arma::umat> > eb(5);

    for(size_t rank=0; rank<=4; rank++)
    {
        ea[rank] = excitations(nocca, nmo, rank, max_virt, max_exc);
        eb[rank] = excitations(noccb, nmo, rank, max_virt, max_exc);

        std::cout << "alpha " << rank_name(rank) << " excitations: "
                  << ea[rank].size() << std::endl;
        std::cout << "beta  " << rank_name(rank) << " excitations: "
                  << eb[rank].size() << std::endl;
    }

    // Build spin-resolved excitation classes
    std::vector<excitation_class> classes = excitation_classes();

    // Evaluate every ordered pair of classes up to mixed quad/mixed quad
    double checksum = 0.0;
    std::vector<benchmark_record> records;

    for(size_t i=0; i<classes.size(); i++)
    for(size_t j=0; j<classes.size(); j++)
        run_class_pair(mb, task, classes[i], classes[j], ea, eb, checksum, records);

    // Report checksum and timing summary
    print_benchmark_summary(bench_name, task, records, checksum);

    if(!std::isfinite(checksum)) return 1;

    return 0;
}

#endif // LIBGNME_TEST_WICK_BENCH_COMMON_H
