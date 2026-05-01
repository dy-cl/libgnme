#ifndef LIBGNME_WICK_EVAL_ONE_BODY_H
#define LIBGNME_WICK_EVAL_ONE_BODY_H

#include <armadillo>
#include <cstdint>
#include <libgnme/utils/linalg.h>

#include "helpers.h"
#include "prepare.h"

namespace libgnme {
namespace wick_eval {

/** \brief Evaluate one-body matrix element for nz = 0 and zero excitations.
    \tparam Tc Matrix element type.
    \param F Output one-body matrix element.
    \param F0 Zeroth-order one-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_l0(Tc &F, const arma::Col<Tc> &F0)
{
    F = F0(0);
}

/** \brief Evaluate one-body matrix element for nz = 0 and one excitation.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param X Lower-triangular contractions.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_l1(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX)
{
    const size_t r0 = rows(0);
    const size_t c0 = cols(0);

    F = X(0)(r0,c0) * F0(0) - XFX(0,0)(r0,c0);
}

/** \brief Evaluate one-body matrix element for nz = 0 and two excitations.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_l2(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX)
{
    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    const Tc detD = det2(D);
    const arma::Mat<Tc> &Fmat = XFX(0,0);

    const size_t r0 = rows(0);
    const size_t r1 = rows(1);
    const size_t c0 = cols(0);
    const size_t c1 = cols(1);

    const Tc det_c0 = Fmat(r0,c0) * D(1,1) - D(0,1) * Fmat(r1,c0);
    const Tc det_c1 = D(0,0) * Fmat(r1,c1) - Fmat(r0,c1) * D(1,0);

    F = detD * F0(0) - det_c0 - det_c1;
}

/** \brief Evaluate one-body matrix element for nz = 0 and arbitrary excitation rank.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX)
{
    const size_t nex = rows.n_elem;

    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    arma::Mat<Tc> Fmat;
    build_mat(XFX(0,0), rows, cols, Fmat);

    Tc detD;
    size_t nzero;
    arma::Mat<Tc> adjD;
    adjoint_matrix(D, adjD, detD, nzero);
    adjD = adjD.t();

    F = F0(0) * detD;

    for(size_t i=0; i<nex; i++)
    {
        arma::Col<Tc> v1(Fmat.colptr(i), nex, false, true);
        arma::Col<Tc> v2(D.colptr(i), nex, false, true);
        arma::Col<Tc> a(adjD.colptr(i), nex, false, true);

        F -= (detD + arma::dot(v1-v2, a));
    }
}

/** \brief Dispatch one-body matrix element for nz = 0.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX)
{
    const size_t nex = rows.n_elem;

    if(nex == 0)
    {
        one_body_m0_l0(F, F0);
        return;
    }

    if(nex == 1)
    {
        one_body_m0_l1(rows, cols, F, X, F0, XFX);
        return;
    }

    if(nex == 2)
    {
        one_body_m0_l2(rows, cols, F, X, Y, F0, XFX);
        return;
    }

    one_body_m0_gen(rows, cols, F, X, Y, F0, XFX);
}

/** \brief Evaluate one-body matrix element for the generic nz > 0 case.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const size_t &nz,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX)
{
    const size_t nex = rows.n_elem;
    const size_t dim = (nz > 0) ? 2 : 1;

    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    arma::Mat<Tc> Db;
    build_det(X(1), Y(1), rows, cols, Db);

    arma::field<arma::Mat<Tc> > Ftmp(dim,dim);
    for(size_t i=0; i<dim; i++)
    for(size_t j=0; j<dim; j++)
        build_mat(XFX(i,j), rows, cols, Ftmp(i,j));

    arma::Mat<Tc> Dtmp;

    for_each_m_combination(nex+1, nz, [&](uint64_t bits) {
        const size_t m0 = bit(bits, 0);

        mix_det(D, Db, bits, 1, Dtmp);

        Tc detDtmp;
        size_t nzero;
        arma::Mat<Tc> adjDtmp;
        adjoint_matrix(Dtmp, adjDtmp, detDtmp, nzero);
        adjDtmp = adjDtmp.t();

        F += F0(m0) * detDtmp;

        for(size_t i=0; i<nex; i++)
        {
            const size_t mi = bit(bits, i+1);

            arma::Col<Tc> v1(Ftmp(m0,mi).colptr(i), nex, false, true);
            arma::Col<Tc> v2(Dtmp.colptr(i), nex, false, true);
            arma::Col<Tc> a(adjDtmp.colptr(i), nex, false, true);

            F -= (detDtmp + arma::dot(v1-v2, a));
        }
    });
}

/** \brief Evaluate one-body same-spin contribution using Wick contractions.
    \tparam Tc Matrix element type.
    \param xhp Particle-hole indices for bra state.
    \param whp Particle-hole indices for ket state.
    \param F Output one-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \param wshift Ket orbital index shift.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void spin_one_body(
    arma::umat xhp, arma::umat whp,
    Tc &F,
    const size_t &nz,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX,
    const size_t &wshift)
{
    F = Tc(0.0);

    const size_t nx = xhp.n_rows;
    const size_t nw = whp.n_rows;
    const size_t nex = nx + nw;

    if(nz > nex + 1) return;

    whp += wshift;

    arma::uvec rows, cols;
    indices(xhp, whp, rows, cols);

    if(nz == 0)
    {
        one_body_m0(rows, cols, F, X, Y, F0, XFX);
        return;
    }

    one_body_gen(rows, cols, F, nz, X, Y, F0, XFX);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_ONE_BODY_H
