#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <armadillo>

#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/wick/wick_uscf.h>
#include "testing.h"

using namespace libgnme;

struct timer
{
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

    timer(const std::string &_name) :
        name(_name), start(std::chrono::high_resolution_clock::now()) {}

    ~timer()
    {
        const auto stop = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(stop - start).count();
        std::cout << name << ": " << std::fixed << std::setprecision(6)
                  << dt << " s" << std::endl;
    }
};

struct eval_case
{
    arma::umat xa;
    arma::umat xb;
    arma::umat wa;
    arma::umat wb;
};

void combinations_rec(
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

std::vector<std::vector<size_t> > combinations(const size_t stop, const size_t rank)
{
    std::vector<size_t> cur;
    std::vector<std::vector<size_t> > out;

    combinations_rec(0, stop, rank, cur, out);

    return out;
}

std::vector<arma::umat> excitations(
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

std::vector<eval_case> make_cases(
    const std::vector<arma::umat> &xa,
    const std::vector<arma::umat> &xb,
    const std::vector<arma::umat> &wa,
    const std::vector<arma::umat> &wb,
    const size_t max_cases)
{
    std::vector<eval_case> cases;

    for(size_t i=0; i<xa.size(); i++)
    for(size_t j=0; j<xb.size(); j++)
    for(size_t k=0; k<wa.size(); k++)
    for(size_t l=0; l<wb.size(); l++)
    {
        eval_case c;
        c.xa = xa[i];
        c.xb = xb[j];
        c.wa = wa[k];
        c.wb = wb[l];

        cases.push_back(c);

        if(cases.size() >= max_cases)
            return cases;
    }

    return cases;
}

template<typename Wick>
void run_cases(
    const std::string &name,
    Wick &mb,
    const std::vector<eval_case> cases,
    double &checksum)
{
    double swick = 0.0;
    double fwick = 0.0;

    std::cout << name << " cases: " << cases.size() << std::endl;

    {
        timer t(name);

        for(size_t i=0; i<cases.size(); i++)
        {

            arma::umat xa = cases[i].xa;
            arma::umat xb = cases[i].xb;
            arma::umat wa = cases[i].wa;
            arma::umat wb = cases[i].wb;

            mb.evaluate(xa, xb, wa, wb, swick, fwick);

            checksum += swick + fwick;
        }
    }
}

int main()
{
    const char *testcase = "h2o_6-31g";

    std::string fname_nmo = testcase + std::string("/nelec.txt");
    std::string fname_c   = testcase + std::string("/coeff.txt");
    std::string fname_ov  = testcase + std::string("/ovls.txt");
    std::string fname_II  = testcase + std::string("/teis.bin");

    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0;

    std::ifstream nelec_file(fname_nmo);
    if(!nelec_file)
    {
        std::cerr << "Could not open " << fname_nmo << std::endl;
        return 1;
    }
    nelec_file >> nbsf >> nocca >> noccb >> nact;
    nelec_file.close();

    nmo = nbsf;

    assert(nmo   > 0);
    assert(nbsf  > 0);
    assert(nocca > 0);
    assert(noccb > 0);

    arma::mat S;
    if(!S.load(fname_ov, arma::raw_ascii))
    {
        std::cerr << "Could not open " << fname_ov << std::endl;
        return 1;
    }
    assert(S.n_rows == nbsf);
    assert(S.n_cols == nbsf);

    arma::mat II;
    if(!II.load(fname_II, arma::raw_binary))
    {
        std::cerr << "Could not open " << fname_II << std::endl;
        return 1;
    }
    assert(II.n_elem == nbsf*nbsf*nbsf*nbsf);
    II.reshape(nbsf*nbsf, nbsf*nbsf);

    arma::mat Cread;
    if(!Cread.load(fname_c, arma::raw_ascii))
    {
        std::cerr << "Could not open " << fname_c << std::endl;
        return 1;
    }
    assert(Cread.n_rows == nbsf);
    assert(Cread.n_cols == 2*nmo);

    arma::mat Cx(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cx_a(Cx.colptr(0), nbsf, nmo, false, true);
    arma::mat Cx_b(Cx.colptr(nmo), nbsf, nmo, false, true);

    arma::mat Cw(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cw_a(Cw.colptr(0), nbsf, nmo, false, true);
    arma::mat Cw_b(Cw.colptr(nmo), nbsf, nmo, false, true);

    Cx_a = Cread.cols(0,nmo-1);
    Cw_b = Cread.cols(0,nmo-1);
    Cx_b = Cread.cols(nmo,2*nmo-1);
    Cw_a = Cread.cols(nmo,2*nmo-1);

    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    wick_uscf<double,double,double> mb(orbs_a, orbs_b);

    {
        timer t("mb.add_two_body");
        mb.add_two_body(II);
    }

    const size_t max_virt = 6;
    const size_t max_exc  = 500;

    std::vector<arma::umat> ea0 = excitations(nocca, nmo, 0, max_virt, max_exc);
    std::vector<arma::umat> ea1 = excitations(nocca, nmo, 1, max_virt, max_exc);
    std::vector<arma::umat> ea2 = excitations(nocca, nmo, 2, max_virt, max_exc);
    std::vector<arma::umat> ea3 = excitations(nocca, nmo, 3, max_virt, max_exc);
    std::vector<arma::umat> ea4 = excitations(nocca, nmo, 4, max_virt, max_exc);

    std::vector<arma::umat> eb0 = excitations(noccb, nmo, 0, max_virt, max_exc);
    std::vector<arma::umat> eb1 = excitations(noccb, nmo, 1, max_virt, max_exc);
    std::vector<arma::umat> eb2 = excitations(noccb, nmo, 2, max_virt, max_exc);
    std::vector<arma::umat> eb3 = excitations(noccb, nmo, 3, max_virt, max_exc);
    std::vector<arma::umat> eb4 = excitations(noccb, nmo, 4, max_virt, max_exc);

    double checksum = 0.0;

    run_cases("mb.evaluate ref/ref",
              mb, make_cases(ea0, eb0, ea0, eb0, 1), checksum);

    run_cases("mb.evaluate alpha ref/single",
              mb, make_cases(ea1, eb0, ea0, eb0, 5000), checksum);

    run_cases("mb.evaluate beta ref/single",
              mb, make_cases(ea0, eb1, ea0, eb0, 5000), checksum);

    run_cases("mb.evaluate alpha single/single",
              mb, make_cases(ea1, eb0, ea1, eb0, 10000), checksum);

    run_cases("mb.evaluate beta single/single",
              mb, make_cases(ea0, eb1, ea0, eb1, 10000), checksum);

    run_cases("mb.evaluate mixed single/single",
              mb, make_cases(ea1, eb0, ea0, eb1, 10000), checksum);

    run_cases("mb.evaluate alpha single/double",
              mb, make_cases(ea1, eb0, ea2, eb0, 20000), checksum);

    run_cases("mb.evaluate beta single/double",
              mb, make_cases(ea0, eb1, ea0, eb2, 20000), checksum);

    run_cases("mb.evaluate mixed single/double",
              mb, make_cases(ea1, eb0, ea1, eb1, 20000), checksum);

    run_cases("mb.evaluate alpha double/double",
              mb, make_cases(ea2, eb0, ea2, eb0, 30000), checksum);

    run_cases("mb.evaluate beta double/double",
              mb, make_cases(ea0, eb2, ea0, eb2, 30000), checksum);

    run_cases("mb.evaluate mixed double/double",
              mb, make_cases(ea2, eb1, ea1, eb2, 30000), checksum);

    run_cases("mb.evaluate alpha triple/single",
              mb, make_cases(ea3, eb0, ea1, eb0, 30000), checksum);

    run_cases("mb.evaluate beta triple/single",
              mb, make_cases(ea0, eb3, ea0, eb1, 30000), checksum);

    run_cases("mb.evaluate alpha triple/double",
              mb, make_cases(ea3, eb0, ea2, eb0, 30000), checksum);

    run_cases("mb.evaluate beta triple/double",
              mb, make_cases(ea0, eb3, ea0, eb2, 30000), checksum);

    run_cases("mb.evaluate mixed triple/double",
              mb, make_cases(ea3, eb1, ea1, eb2, 30000), checksum);

    run_cases("mb.evaluate alpha quadruple/double",
              mb, make_cases(ea4, eb0, ea2, eb0, 30000), checksum);

    run_cases("mb.evaluate beta quadruple/double",
              mb, make_cases(ea0, eb4, ea0, eb2, 30000), checksum);

    run_cases("mb.evaluate mixed high-rank",
              mb, make_cases(ea2, eb2, ea3, eb1, 30000), checksum);

    std::cout << "checksum: " << std::setprecision(16) << checksum << std::endl;

    if(!std::isfinite(checksum)) return 1;

    return 0;
}
