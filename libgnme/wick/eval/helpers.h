#ifndef LIBGNME_WICK_EVAL_HELPERS_H
#define LIBGNME_WICK_EVAL_HELPERS_H

#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <libgnme/utils/linalg.h>

namespace libgnme {
namespace wick_eval {

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

/** \brief Return determinant of a 2x2 matrix from scalar entries.
    \tparam T Matrix element type.
    \param a00 Matrix element.
    \param a01 Matrix element.
    \param a10 Matrix element.
    \param a11 Matrix element.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det2_scalar(
    const T a00, const T a01,
    const T a10, const T a11)
{
    return a00 * a11 - a01 * a10;
}

/** \brief Return determinant of a 3x3 matrix from scalar entries.
    \tparam T Matrix element type.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det3_scalar(
    const T a00, const T a01, const T a02,
    const T a10, const T a11, const T a12,
    const T a20, const T a21, const T a22)
{
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
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

/** \brief Compute determinant and cofactor matrix for a 1x1 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output cofactor matrix.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose1(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    cof.set_size(1,1);
    cof(0,0) = T(1.0);

    return A(0,0);
}

/** \brief Compute determinant and cofactor matrix for a 2x2 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output cofactor matrix.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose2(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    const T a00 = A(0,0);
    const T a01 = A(0,1);
    const T a10 = A(1,0);
    const T a11 = A(1,1);

    const T det = det2_scalar(a00, a01, a10, a11);

    cof.set_size(2,2);
    cof(0,0) =  a11;
    cof(0,1) = -a10;
    cof(1,0) = -a01;
    cof(1,1) =  a00;

    return det;
}

/** \brief Compute determinant and cofactor matrix for a 3x3 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output cofactor matrix.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose3(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    const T a00 = A(0,0);
    const T a01 = A(0,1);
    const T a02 = A(0,2);

    const T a10 = A(1,0);
    const T a11 = A(1,1);
    const T a12 = A(1,2);

    const T a20 = A(2,0);
    const T a21 = A(2,1);
    const T a22 = A(2,2);

    const T det = det3_scalar(
        a00, a01, a02,
        a10, a11, a12,
        a20, a21, a22);

    cof.set_size(3,3);

    cof(0,0) =  det2_scalar(a11, a12, a21, a22);
    cof(0,1) = -det2_scalar(a10, a12, a20, a22);
    cof(0,2) =  det2_scalar(a10, a11, a20, a21);

    cof(1,0) = -det2_scalar(a01, a02, a21, a22);
    cof(1,1) =  det2_scalar(a00, a02, a20, a22);
    cof(1,2) = -det2_scalar(a00, a01, a20, a21);

    cof(2,0) =  det2_scalar(a01, a02, a11, a12);
    cof(2,1) = -det2_scalar(a00, a02, a10, a12);
    cof(2,2) =  det2_scalar(a00, a01, a10, a11);

    return det;
}

/** \brief Compute determinant and cofactor matrix for a 4x4 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output cofactor matrix.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose4(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    const T det = det4(A);

    cof.set_size(4,4);

    for(size_t i=0; i<4; i++)
    for(size_t j=0; j<4; j++)
    {
        size_t r[3];
        size_t c[3];

        size_t ri = 0;
        for(size_t rr=0; rr<4; rr++)
        {
            if(rr == i) continue;
            r[ri] = rr;
            ri++;
        }

        size_t ci = 0;
        for(size_t cc=0; cc<4; cc++)
        {
            if(cc == j) continue;
            c[ci] = cc;
            ci++;
        }

        const T m00 = A(r[0],c[0]);
        const T m01 = A(r[0],c[1]);
        const T m02 = A(r[0],c[2]);

        const T m10 = A(r[1],c[0]);
        const T m11 = A(r[1],c[1]);
        const T m12 = A(r[1],c[2]);

        const T m20 = A(r[2],c[0]);
        const T m21 = A(r[2],c[1]);
        const T m22 = A(r[2],c[2]);

        const T minor_det = det3_scalar(
            m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22);

        cof(i,j) = (((i + j) & 1) == 0) ? minor_det : -minor_det;
    }

    return det;
}

/** \brief Compute determinant and cofactor matrix using explicit small-rank formulae.
    \tparam T Matrix element type.
    \param A Input square matrix.
    \param cof Output cofactor matrix.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    assert(A.n_rows == A.n_cols);

    if(A.n_rows == 0)
    {
        cof.set_size(0,0);
        return T(1.0);
    }

    if(A.n_rows == 1) return adjugate_transpose1(A, cof);
    if(A.n_rows == 2) return adjugate_transpose2(A, cof);
    if(A.n_rows == 3) return adjugate_transpose3(A, cof);
    if(A.n_rows == 4) return adjugate_transpose4(A, cof);

    T det;
    size_t nzero;
    arma::Mat<T> Acopy = A;
    adjoint_matrix(Acopy, cof, det, nzero);
    cof = cof.t();

    return det;
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

/** \brief Convert two zero-branch selectors into a branch-pair index.
\param m0 First branch selector.
\param m1 Second branch selector.
\return Branch-pair index.
\ingroup gnme_wick
 **/
inline size_t two_body_pair(const size_t m0, const size_t m1)
{
    return 2 * m0 + m1;
}

/** \brief Canonicalise same-spin branch-pair indices using pair-exchange symmetry.
    \param p First branch-pair index.
    \param q Second branch-pair index.
    \param transpose Whether the requested block is stored transposed.
    \ingroup gnme_wick
 **/
inline void two_body_same_canonical(size_t &p, size_t &q, bool &transpose)
{
    transpose = (p > q);

    if(transpose)
        std::swap(p, q);
}

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
    
    const Tc det_minor = adjugate_transpose(Dminor, cof_minor);
    f(lm1, Dminor, cof_minor, det_minor);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_HELPERS_H
