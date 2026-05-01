#ifndef LIBGNME_WICK_EVAL_OVERLAP_H
#define LIBGNME_WICK_EVAL_OVERLAP_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "prepare.h"

namespace libgnme {
namespace wick_eval {

/** \brief Evaluate same-spin overlap contribution using Wick contractions.
    \tparam Tc Matrix element type.
    \param xhp Particle-hole indices for bra state.
    \param whp Particle-hole indices for ket state.
    \param S Output overlap.
    \param nz Number of zero-overlap orbital pairs.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param wshift Ket orbital index shift.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void spin_overlap(
    arma::umat xhp, arma::umat whp,
    Tc &S,
    const size_t &nz,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const size_t &wshift)
{
    S = Tc(0.0);

    const size_t nx = xhp.n_rows;
    const size_t nw = whp.n_rows;
    const size_t nex = nx + nw;

    if(nz > nex) return;

    whp += wshift;

    arma::uvec rows, cols;
    indices(xhp, whp, rows, cols);

    if(nex == 0)
    {
        S = (nz == 0) ? Tc(1.0) : Tc(0.0);
        return;
    }

    if(nex == 1)
    {
        S = X(nz)(rows(0),cols(0));
        return;
    }

    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    if(nz == 0)
    {
        S = det(D);
        return;
    }

    arma::Mat<Tc> Db;
    build_det(X(1), Y(1), rows, cols, Db);

    if(nz == nex)
    {
        S = det(Db);
        return;
    }

    arma::Mat<Tc> Dtmp;

    for_each_m_combination(nex, nz, [&](uint64_t bits) {
        mix_det(D, Db, bits, 0, Dtmp);
        S += det(Dtmp);
    });
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_OVERLAP_H
