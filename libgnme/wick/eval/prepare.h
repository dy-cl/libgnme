#ifndef LIBGNME_WICK_EVAL_PREPARE_H
#define LIBGNME_WICK_EVAL_PREPARE_H

#include <armadillo>
#include <cstdint>
#include <vector>

#include "helpers.h"

namespace libgnme {
namespace wick_eval {

/** \brief Build Wick determinant row and column index arrays.
    \param xhp Particle-hole indices for the bra excitation.
    \param whp Particle-hole indices for the ket excitation.
    \param rows Output row indices.
    \param cols Output column indices.
    \ingroup gnme_wick
 **/
inline void indices(
    const arma::umat &xhp,
    const arma::umat &whp,
    arma::uvec &rows,
    arma::uvec &cols)
{
    const size_t nx = xhp.n_rows;
    const size_t nw = whp.n_rows;
    const size_t nex = nx + nw;

    rows.set_size(nex);
    cols.set_size(nex);

    for(size_t i=0; i<nx; i++)
    {
        rows(i) = xhp(i,1);
        cols(i) = xhp(i,0);
    }

    for(size_t i=0; i<nw; i++)
    {
        rows(nx+i) = whp(i,0);
        cols(nx+i) = whp(i,1);
    }
}

/** \brief Build a Wick determinant matrix from lower/upper contraction branches.
    \tparam T Matrix element type.
    \param X Lower-triangular contraction matrix.
    \param Y Upper-triangular contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param D Output determinant matrix.
    \ingroup gnme_wick
 **/
template<typename T>
inline void build_det(
    const arma::Mat<T> &X,
    const arma::Mat<T> &Y,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<T> &D)
{
    const size_t n = rows.n_elem;

    D.set_size(n,n);

    for(size_t j=0; j<n; j++)
    for(size_t i=0; i<n; i++)
        D(i,j) = (i >= j) ? X(rows(i),cols(j)) : Y(rows(i),cols(j));
}

/** \brief Build a dense contraction submatrix.
    \tparam T Matrix element type.
    \param A Full contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param B Output submatrix.
    \ingroup gnme_wick
 **/
template<typename T>
inline void build_mat(
    const arma::Mat<T> &A,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<T> &B)
{
    const size_t n = rows.n_elem;

    B.set_size(n,n);

    for(size_t j=0; j<n; j++)
    for(size_t i=0; i<n; i++)
        B(i,j) = A(rows(i),cols(j));
}

/** \brief Mix columns from two determinant matrices according to a vector of selectors.
    \tparam T Matrix element type.
    \param D Non-zero-overlap determinant branch.
    \param Db Zero-overlap determinant branch.
    \param m Column selector vector. Zero selects D; non-zero selects Db.
    \param Dtmp Output mixed determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline void mix_det(
    const arma::Mat<T> &D,
    const arma::Mat<T> &Db,
    const std::vector<size_t> &m,
    arma::Mat<T> &Dtmp)
{
    const size_t n = D.n_rows;

    Dtmp.set_size(n,n);

    for(size_t j=0; j<n; j++)
    {
        const arma::Mat<T> &src = m[j] ? Db : D;

        for(size_t i=0; i<n; i++)
            Dtmp(i,j) = src(i,j);
    }
}

/** \brief Mix columns from two determinant matrices according to a bitstring.
    \tparam T Matrix element type.
    \param D Non-zero-overlap determinant branch.
    \param Db Zero-overlap determinant branch.
    \param bits Column selector bitstring.
    \param offset Offset into bitstring for determinant columns.
    \param Dtmp Output mixed determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline void mix_det(
    const arma::Mat<T> &D,
    const arma::Mat<T> &Db,
    const uint64_t bits,
    const size_t offset,
    arma::Mat<T> &Dtmp)
{
    const size_t n = D.n_rows;

    Dtmp.set_size(n,n);

    for(size_t j=0; j<n; j++)
    {
        const arma::Mat<T> &src = bit(bits,j+offset) ? Db : D;

        for(size_t i=0; i<n; i++)
            Dtmp(i,j) = src(i,j);
    }
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_PREPARE_H
