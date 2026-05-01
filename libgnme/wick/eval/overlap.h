#ifndef LIBGNME_WICK_EVAL_OVERLAP_H
#define LIBGNME_WICK_EVAL_OVERLAP_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "prepare.h"

namespace libgnme {
namespace wick_eval {

/** \brief Evaluate a one-column overlap determinant.
    \tparam Tc Matrix element type.
    \param X Lower-triangular contractions.
    \param rows Row indices.
    \param cols Column indices.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_l1(
    const arma::Mat<Tc> &X,
    const arma::uvec &rows, const arma::uvec &cols)
{
    return X(rows(0),cols(0));
}

/** \brief Evaluate a two-column overlap determinant.
    \tparam Tc Matrix element type.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param rows Row indices.
    \param cols Column indices.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_l2(
    const arma::Mat<Tc> &X, const arma::Mat<Tc> &Y,
    const arma::uvec &rows, const arma::uvec &cols)
{
    const size_t r0 = rows(0);
    const size_t r1 = rows(1);
    const size_t c0 = cols(0);
    const size_t c1 = cols(1);

    const Tc a00 = X(r0,c0);
    const Tc a01 = Y(r0,c1);
    const Tc a10 = X(r1,c0);
    const Tc a11 = X(r1,c1);

    return a00 * a11 - a01 * a10;
}

/** \brief Evaluate a three-column overlap determinant.
    \tparam Tc Matrix element type.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param rows Row indices.
    \param cols Column indices.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_l3(
    const arma::Mat<Tc> &X, const arma::Mat<Tc> &Y,
    const arma::uvec &rows, const arma::uvec &cols)
{
    const size_t r0 = rows(0);
    const size_t r1 = rows(1);
    const size_t r2 = rows(2);
    const size_t c0 = cols(0);
    const size_t c1 = cols(1);
    const size_t c2 = cols(2);

    const Tc a00 = X(r0,c0);
    const Tc a01 = Y(r0,c1);
    const Tc a02 = Y(r0,c2);

    const Tc a10 = X(r1,c0);
    const Tc a11 = X(r1,c1);
    const Tc a12 = Y(r1,c2);

    const Tc a20 = X(r2,c0);
    const Tc a21 = X(r2,c1);
    const Tc a22 = X(r2,c2);

    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
}

/** \brief Evaluate an overlap determinant for one contraction branch.
    \tparam Tc Matrix element type.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param rows Row indices.
    \param cols Column indices.
    \return Overlap determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_branch(
    const arma::Mat<Tc> &X, const arma::Mat<Tc> &Y,
    const arma::uvec &rows, const arma::uvec &cols)
{
    const size_t nex = rows.n_elem;

    if(nex == 0) return Tc(1.0);
    if(nex == 1) return overlap_l1(X, rows, cols);
    if(nex == 2) return overlap_l2(X, Y, rows, cols);
    if(nex == 3) return overlap_l3(X, Y, rows, cols);

    arma::Mat<Tc> D;
    build_det(X, Y, rows, cols, D);
    return det(D);
}

/** \brief Evaluate same-spin overlap for the nz = 0 branch.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \return Overlap matrix element.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_m0(
    const arma::uvec &rows, const arma::uvec &cols,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y)
{
    return overlap_branch(X(0), Y(0), rows, cols);
}

/** \brief Evaluate same-spin overlap when all determinant columns are zero-replacement columns.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \return Overlap matrix element.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_ml(
    const arma::uvec &rows, const arma::uvec &cols,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y)
{
    return overlap_branch(X(1), Y(1), rows, cols);
}

/** \brief Evaluate same-spin overlap for the generic mixed-column case.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param nz Number of zero-overlap orbital pairs.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \return Overlap matrix element.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc overlap_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    const size_t &nz,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y)
{
    const size_t nex = rows.n_elem;

    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    arma::Mat<Tc> Db;
    build_det(X(1), Y(1), rows, cols, Db);

    Tc S = Tc(0.0);
    arma::Mat<Tc> Dtmp;

    for_each_m_combination(nex, nz, [&](uint64_t bits) {
        mix_det(D, Db, bits, 0, Dtmp);
        S += det(Dtmp);
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

    whp += wshift;

    arma::uvec rows, cols;
    indices(xhp, whp, rows, cols);

    if(nz == 0)
    {
        S = overlap_m0(rows, cols, X, Y);
        return;
    }

    if(nz == nex)
    {
        S = overlap_ml(rows, cols, X, Y);
        return;
    }

    S = overlap_gen(rows, cols, nz, X, Y);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_OVERLAP_H
