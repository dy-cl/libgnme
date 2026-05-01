#ifndef LIBGNME_WICK_EVAL_HELPERS_H
#define LIBGNME_WICK_EVAL_HELPERS_H

#include <armadillo>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace libgnme {
namespace wick_eval {

/** \brief Map an index in a minor matrix back to the full matrix.
    \param midx Index in the minor matrix.
    \param removed Removed row or column index in the full matrix.
    \return Corresponding index in the full matrix.
    \ingroup gnme_wick
 **/
inline size_t minor_to_full(const size_t midx, const size_t removed)
{
    return (midx < removed) ? midx : midx + 1;
}

/** \brief Compute determinant correction from replacing one column.
    \tparam Tc Matrix element type.
    \tparam Fn Replacement-column callback type.
    \param old Original determinant matrix.
    \param cof Cofactor matrix matching old.
    \param col Column to replace.
    \param new_at Callback returning replacement value for row r.
    \return Cofactor contraction correction.
    \ingroup gnme_wick
 **/
template<typename Tc, typename Fn>
inline Tc column_replacement_correction(
    const arma::Mat<Tc> &old,
    const arma::Mat<Tc> &cof,
    const size_t col,
    Fn &&new_at)
{
    const size_t n = old.n_rows;
    Tc correction = Tc(0.0);

    for(size_t r=0; r<n; r++)
        correction += (new_at(r) - old(r,col)) * cof(r,col);

    return correction;
}

/** \brief Build a determinant minor and its cofactor matrix.
    \tparam Tc Matrix element type.
    \tparam Fn Callback type.
    \param D Full determinant matrix.
    \param row_rm Removed row.
    \param col_rm Removed column.
    \param Dminor Output minor determinant.
    \param cof_minor Output minor cofactor matrix.
    \param f Callback receiving minor size, minor determinant, cofactor matrix, and determinant.
    \ingroup gnme_wick
 **/
template<typename Tc, typename Fn>
inline void minor_adjt(
    const arma::Mat<Tc> &D,
    const size_t row_rm,
    const size_t col_rm,
    arma::Mat<Tc> &Dminor,
    arma::Mat<Tc> &cof_minor,
    Fn &&f)
{
    const size_t l = D.n_rows;
    const size_t lm1 = l - 1;

    Dminor.set_size(lm1,lm1);

    for(size_t j=0; j<lm1; j++)
    for(size_t i=0; i<lm1; i++)
        Dminor(i,j) = D(minor_to_full(i,row_rm), minor_to_full(j,col_rm));

    Tc det_minor;
    size_t nzero;
    adjoint_matrix(Dminor, cof_minor, det_minor, nzero);
    cof_minor = cof_minor.t();

    f(lm1, Dminor, cof_minor, det_minor);
}

/** \brief Extract a bit from a bitstring.
    \param bits Bitstring.
    \param k Bit index.
    \return Selected bit.
    \ingroup gnme_wick
 **/
inline size_t bit(const uint64_t bits, const size_t k)
{
    return (bits >> k) & 1ULL;
}

/** \brief Call a function for every bitstring with a fixed number of set bits.
    \tparam Fn Callback type.
    \param l Bitstring length.
    \param m Number of set bits.
    \param f Callback function.
    \ingroup gnme_wick
 **/
template<typename Fn>
inline void for_each_m_combination(const size_t l, const size_t m, Fn &&f)
{
    assert(l < 64);

    if(m > l) return;

    if(m == 0)
    {
        f(uint64_t(0));
        return;
    }

    if(m == l)
    {
        f((uint64_t(1) << l) - uint64_t(1));
        return;
    }

    const uint64_t limit = uint64_t(1) << l;
    uint64_t x = (uint64_t(1) << m) - uint64_t(1);

    while(x < limit)
    {
        f(x);

        const uint64_t c = x & -x;
        const uint64_t r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
}

/** \brief Return determinant of a 2x2 matrix.
    \tparam T Matrix element type.
    \param A Matrix data.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det2(const arma::Mat<T> &A)
{
    return A(0,0) * A(1,1) - A(0,1) * A(1,0);
}

/** \brief Return determinant of a 3x3 matrix.
    \tparam T Matrix element type.
    \param A Matrix data.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det3(const arma::Mat<T> &A)
{
    return A(0,0) * (A(1,1) * A(2,2) - A(1,2) * A(2,1))
         - A(0,1) * (A(1,0) * A(2,2) - A(1,2) * A(2,0))
         + A(0,2) * (A(1,0) * A(2,1) - A(1,1) * A(2,0));
}

/** \brief Return determinant of a 4x4 matrix.
    \tparam T Matrix element type.
    \param A Matrix data.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det4(const arma::Mat<T> &A)
{
    const T m0 = A(1,1) * (A(2,2) * A(3,3) - A(2,3) * A(3,2))
               - A(1,2) * (A(2,1) * A(3,3) - A(2,3) * A(3,1))
               + A(1,3) * (A(2,1) * A(3,2) - A(2,2) * A(3,1));

    const T m1 = A(1,0) * (A(2,2) * A(3,3) - A(2,3) * A(3,2))
               - A(1,2) * (A(2,0) * A(3,3) - A(2,3) * A(3,0))
               + A(1,3) * (A(2,0) * A(3,2) - A(2,2) * A(3,0));

    const T m2 = A(1,0) * (A(2,1) * A(3,3) - A(2,3) * A(3,1))
               - A(1,1) * (A(2,0) * A(3,3) - A(2,3) * A(3,0))
               + A(1,3) * (A(2,0) * A(3,1) - A(2,1) * A(3,0));

    const T m3 = A(1,0) * (A(2,1) * A(3,2) - A(2,2) * A(3,1))
               - A(1,1) * (A(2,0) * A(3,2) - A(2,2) * A(3,0))
               + A(1,2) * (A(2,0) * A(3,1) - A(2,1) * A(3,0));

    return A(0,0) * m0 - A(0,1) * m1 + A(0,2) * m2 - A(0,3) * m3;
}

/** \brief Return determinant using explicit small-rank formula where possible.
    \tparam T Matrix element type.
    \param A Matrix data.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det(const arma::Mat<T> &A)
{
    assert(A.n_rows == A.n_cols);

    if(A.n_rows == 0) return T(1.0);
    if(A.n_rows == 1) return A(0,0);
    if(A.n_rows == 2) return det2(A);
    if(A.n_rows == 3) return det3(A);
    if(A.n_rows == 4) return det4(A);

    return arma::det(A);
}

/** \brief Compute determinant correction from replacing one column.
    \tparam T Matrix element type.
    \param adj Adjugate matrix of the base determinant.
    \param col Replacement column.
    \param j Replaced column index.
    \return Determinant correction.
    \ingroup gnme_wick
 **/
template<typename T>
inline T column_replacement(
    const arma::Mat<T> &adj,
    const arma::Col<T> &col,
    const size_t j)
{
    T out = T(0.0);

    for(size_t i=0; i<col.n_elem; i++)
        out += adj(j,i) * col(i);

    return out;
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_HELPERS_H
