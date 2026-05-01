#ifndef LIBGNME_WICK_EVAL_OVERLAP_H
#define LIBGNME_WICK_EVAL_OVERLAP_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "prepare.h"
#include "scratch.h"
#include "prepare_scratch.h"

namespace libgnme {
namespace wick_eval {

/** \brief Evaluate a prepared overlap determinant for l = 0.
    \tparam Tc Matrix element type.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_l0()
{
    return Tc(1.0);
}

/** \brief Evaluate a prepared overlap determinant for l = 1.
    \tparam Tc Matrix element type.
    \param D Prepared determinant matrix.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_l1(const arma::Mat<Tc> &D)
{
    return D(0,0);
}

/** \brief Evaluate a prepared overlap determinant for l = 2.
    \tparam Tc Matrix element type.
    \param D Prepared determinant matrix.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_l2(const arma::Mat<Tc> &D)
{
    return det2_scalar(D(0,0), D(0,1), D(1,0), D(1,1));
}

/** \brief Evaluate a prepared overlap determinant for l = 3.
    \tparam Tc Matrix element type.
    \param D Prepared determinant matrix.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_l3(const arma::Mat<Tc> &D)
{
    return det3_scalar(
        D(0,0), D(0,1), D(0,2),
        D(1,0), D(1,1), D(1,2),
        D(2,0), D(2,1), D(2,2));
}

/** \brief Evaluate a prepared overlap determinant.
    \tparam Tc Matrix element type.
    \param D Prepared determinant matrix.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_det(const arma::Mat<Tc> &D)
{
    const size_t nex = D.n_rows;

    if(nex == 0) return overlap_l0<Tc>();
    if(nex == 1) return overlap_l1(D);
    if(nex == 2) return overlap_l2(D);
    if(nex == 3) return overlap_l3(D);

    return det(D);
}

/** \brief Evaluate same-spin overlap for the nz = 0 branch.
    \tparam Tc Matrix element type.
    \param work Same-spin scratch storage.
    \return Overlap matrix element.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_m0(same_scratch<Tc> &work)
{
    return overlap_det(work.det0);
}

/** \brief Evaluate same-spin overlap when all determinant columns are zero-replacement columns.
    \tparam Tc Matrix element type.
    \param work Same-spin scratch storage.
    \return Overlap matrix element.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_ml(same_scratch<Tc> &work)
{
    return overlap_det(work.det1);
}

/** \brief Evaluate same-spin overlap for the generic mixed-column case.
    \tparam Tc Matrix element type.
    \param nex Excitation rank.
    \param nz Number of zero-overlap orbital pairs.
    \param work Same-spin scratch storage.
    \return Overlap matrix element.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_gen(
    const size_t nex,
    const size_t &nz,
    same_scratch<Tc> &work)
{
    Tc S = Tc(0.0);

    mix_dets_same(nex, nz, 0, work, [&](const uint64_t) {
        S += overlap_det(work.det_mix);
    });

    return S;
}

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

    scratch<Tc> &store = local_scratch<Tc>();
    same_scratch<Tc> &work = store.aa;

    prepare_same(xhp, whp, wshift, nz, X, Y, work);

    if(nz == 0)
    {
        S = overlap_m0(work);
        return;
    }

    if(nz == nex)
    {
        S = overlap_ml(work);
        return;
    }

    S = overlap_gen(nex, nz, work);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_OVERLAP_H
