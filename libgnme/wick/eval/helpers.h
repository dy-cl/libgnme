#ifndef LIBGNME_WICK_EVAL_HELPERS_H
#define LIBGNME_WICK_EVAL_HELPERS_H

#include <algorithm>
#include <cassert>
#include <cstdint>

#include <armadillo>
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
inline void for_each_m_combination(
    const size_t l,
    const size_t m,
    Fn &&f)
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

/** \brief Return determinant of a 2-by-2 matrix from scalar entries.
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
    const T a00,
    const T a01,
    const T a10,
    const T a11)
{
    return a00 * a11 - a01 * a10;
}

/** \brief Return determinant of a 3-by-3 matrix from scalar entries.
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

/** \brief Return determinant of a 2-by-2 matrix.
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

/** \brief Return determinant of a 3-by-3 matrix.
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

/** \brief Return determinant of a 4-by-4 matrix.
    \tparam T Matrix element type.
    \param A Matrix data.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det4(const arma::Mat<T> &A)
{
    const T m0 =
        A(1,1) * (A(2,2) * A(3,3) - A(2,3) * A(3,2))
      - A(1,2) * (A(2,1) * A(3,3) - A(2,3) * A(3,1))
      + A(1,3) * (A(2,1) * A(3,2) - A(2,2) * A(3,1));

    const T m1 =
        A(1,0) * (A(2,2) * A(3,3) - A(2,3) * A(3,2))
      - A(1,2) * (A(2,0) * A(3,3) - A(2,3) * A(3,0))
      + A(1,3) * (A(2,0) * A(3,2) - A(2,2) * A(3,0));

    const T m2 =
        A(1,0) * (A(2,1) * A(3,3) - A(2,3) * A(3,1))
      - A(1,1) * (A(2,0) * A(3,3) - A(2,3) * A(3,0))
      + A(1,3) * (A(2,0) * A(3,1) - A(2,1) * A(3,0));

    const T m3 =
        A(1,0) * (A(2,1) * A(3,2) - A(2,2) * A(3,1))
      - A(1,1) * (A(2,0) * A(3,2) - A(2,2) * A(3,0))
      + A(1,2) * (A(2,0) * A(3,1) - A(2,1) * A(3,0));

    return A(0,0) * m0 - A(0,1) * m1 + A(0,2) * m2 - A(0,3) * m3;
}

/** \brief Compute determinant and adjugate transpose for a 1-by-1 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output adjugate transpose. Must have capacity at least 1-by-1.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose1(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    cof(0,0) = T(1.0);
    return A(0,0);
}

/** \brief Compute determinant and adjugate transpose for a 2-by-2 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output adjugate transpose. Must have capacity at least 2-by-2.
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

    const T d = det2_scalar(a00, a01, a10, a11);

    cof(0,0) =  a11;
    cof(0,1) = -a10;
    cof(1,0) = -a01;
    cof(1,1) =  a00;

    return d;
}

/** \brief Compute determinant and adjugate transpose for a 3-by-3 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output adjugate transpose. Must have capacity at least 3-by-3.
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

    const T d = det3_scalar(
        a00, a01, a02,
        a10, a11, a12,
        a20, a21, a22);

    cof(0,0) =  det2_scalar(a11, a12, a21, a22);
    cof(0,1) = -det2_scalar(a10, a12, a20, a22);
    cof(0,2) =  det2_scalar(a10, a11, a20, a21);

    cof(1,0) = -det2_scalar(a01, a02, a21, a22);
    cof(1,1) =  det2_scalar(a00, a02, a20, a22);
    cof(1,2) = -det2_scalar(a00, a01, a20, a21);

    cof(2,0) =  det2_scalar(a01, a02, a11, a12);
    cof(2,1) = -det2_scalar(a00, a02, a10, a12);
    cof(2,2) =  det2_scalar(a00, a01, a10, a11);

    return d;
}

/** \brief Compute determinant and adjugate transpose for a 4-by-4 matrix.
    \tparam T Matrix element type.
    \param A Input matrix.
    \param cof Output adjugate transpose. Must have capacity at least 4-by-4.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose4(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    const T d = det4(A);

    for(size_t i=0; i<4; i++)
    for(size_t j=0; j<4; j++)
    {
        size_t r[3];
        size_t c[3];

        size_t ri = 0;
        for(size_t rr=0; rr<4; rr++)
        {
            if(rr == i) continue;
            r[ri++] = rr;
        }

        size_t ci = 0;
        for(size_t cc=0; cc<4; cc++)
        {
            if(cc == j) continue;
            c[ci++] = cc;
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

    return d;
}

/** \brief Compute determinant and adjugate transpose for an active leading block.
    \tparam T Matrix element type.
    \param A Input matrix. Only the leading n-by-n block is used.
    \param n Active matrix rank.
    \param cof Output adjugate transpose. Must have capacity at least n-by-n.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose(
    const arma::Mat<T> &A,
    const size_t n,
    arma::Mat<T> &cof)
{
    if(n == 0) return T(1.0);
    if(n == 1) return adjugate_transpose1(A, cof);
    if(n == 2) return adjugate_transpose2(A, cof);
    if(n == 3) return adjugate_transpose3(A, cof);
    if(n == 4) return adjugate_transpose4(A, cof);

    arma::Mat<T> tmp(n,n);
    for(size_t j=0; j<n; j++)
    for(size_t i=0; i<n; i++)
        tmp(i,j) = A(i,j);

    T d;
    size_t nzero;
    arma::Mat<T> cof_tmp;
    adjoint_matrix(tmp, cof_tmp, d, nzero);

    for(size_t j=0; j<n; j++)
    for(size_t i=0; i<n; i++)
        cof(i,j) = cof_tmp(j,i);

    return d;
}

/** \brief Compute determinant and adjugate transpose using full matrix size.
    \tparam T Matrix element type.
    \param A Input square matrix.
    \param cof Output adjugate transpose.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T adjugate_transpose(const arma::Mat<T> &A, arma::Mat<T> &cof)
{
    assert(A.n_rows == A.n_cols);
    cof.set_size(A.n_rows, A.n_cols);
    return adjugate_transpose(A, A.n_rows, cof);
}

/** \brief Return determinant of an active leading block.
    \tparam T Matrix element type.
    \param A Matrix data. Only the leading n-by-n block is used.
    \param n Active matrix rank.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det(const arma::Mat<T> &A, const size_t n)
{
    if(n == 0) return T(1.0);
    if(n == 1) return A(0,0);
    if(n == 2) return det2(A);
    if(n == 3) return det3(A);
    if(n == 4) return det4(A);

    arma::Mat<T> tmp(n,n);
    for(size_t j=0; j<n; j++)
    for(size_t i=0; i<n; i++)
        tmp(i,j) = A(i,j);

    return arma::det(tmp);
}

/** \brief Return determinant using full matrix size.
    \tparam T Matrix element type.
    \param A Matrix data.
    \return Determinant.
    \ingroup gnme_wick
 **/
template<typename T>
inline T det(const arma::Mat<T> &A)
{
    assert(A.n_rows == A.n_cols);
    return det(A, A.n_rows);
}

/** \brief Compute determinant correction from replacing one column.
    \tparam T Matrix element type.
    \param adj Adjugate transpose of the base determinant.
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
        out += adj(i,j) * col(i);

    return out;
}

/** \brief Compute determinant correction from replacing one column.
    \tparam Tc Matrix element type.
    \tparam Fn Replacement-column callback type.
    \param A Active determinant matrix.
    \param cof Adjugate transpose of the active determinant.
    \param n Active determinant rank.
    \param col Column to replace.
    \param new_at Callback returning replacement value for row r.
    \return Cofactor contraction correction.
    \ingroup gnme_wick
 **/
template<typename Tc, typename Fn>
inline Tc column_replacement_correction(
    const arma::Mat<Tc> &,
    const arma::Mat<Tc> &cof,
    const size_t n,
    const size_t col,
    Fn &&new_at)
{
    Tc correction = Tc(0.0);

    for(size_t r=0; r<n; r++)
        correction += cof(r,col) * new_at(r);

    return correction;
}

/** \brief Return compact pair index for two branch selectors.
    \param a First branch selector.
    \param b Second branch selector.
    \return Pair index.
    \ingroup gnme_wick
 **/
inline size_t two_body_pair(const size_t a, const size_t b)
{
    return a + b + a * b;
}

/** \brief Canonicalise two same-spin pair indices.
    \param p First pair index.
    \param q Second pair index.
    \param transpose Whether the canonicalised matrix must be transposed.
    \ingroup gnme_wick
 **/
inline void two_body_same_canonical(size_t &p, size_t &q, bool &transpose)
{
    transpose = false;

    if(q < p)
    {
        transpose = true;
        std::swap(p, q);
    }
}

/** \brief Resolved same-spin two-electron intermediate slot.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
struct same_ii_slot
{
    const arma::Mat<Tc> *M;
    size_t nact;
    bool transpose;

    inline Tc get(
        const size_t r0,
        const size_t c0,
        const size_t r1,
        const size_t c1) const
    {
        const size_t col = nact * r0 + c0;
        const size_t row = c1 + nact * r1;
        return transpose ? (*M)(col,row) : (*M)(row,col);
    }
};

/** \brief Resolve same-spin two-electron intermediate field entry once.
    \tparam Tc Matrix element type.
    \param II Same-spin two-electron intermediate field.
    \param nact Total active dimension.
    \param mi First branch selector.
    \param mj Second branch selector.
    \param mk Third branch selector.
    \param ml Fourth branch selector.
    \return Resolved intermediate slot.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline same_ii_slot<Tc> resolve_same_ii_slot(
    const arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    const size_t mi,
    const size_t mj,
    const size_t mk,
    const size_t ml)
{
    size_t p = two_body_pair(mi, mj);
    size_t q = two_body_pair(mk, ml);

    bool transpose;
    two_body_same_canonical(p, q, transpose);

    same_ii_slot<Tc> out;
    out.M = &II(p,q);
    out.nact = nact;
    out.transpose = transpose;
    return out;
}

/** \brief Resolved alpha-beta two-electron intermediate slot.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
struct diff_ii_slot
{
    const arma::Mat<Tc> *M;
    size_t nacta;
    size_t nactb;

    inline Tc get(
        const size_t ar,
        const size_t ac,
        const size_t br,
        const size_t bc) const
    {
        const size_t row = ac + nacta * ar;
        const size_t col = nactb * br + bc;
        return (*M)(row,col);
    }
};

/** \brief Resolve alpha-beta two-electron intermediate field entry once.
    \tparam Tc Matrix element type.
    \param IIab Alpha-beta two-electron intermediate field.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param ma0 Alpha first branch selector.
    \param maj Alpha second branch selector.
    \param mb0 Beta first branch selector.
    \param mbj Beta second branch selector.
    \return Resolved intermediate slot.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline diff_ii_slot<Tc> resolve_diff_ii_slot(
    const arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    const size_t ma0,
    const size_t maj,
    const size_t mb0,
    const size_t mbj)
{
    diff_ii_slot<Tc> out;
    out.M = &IIab(two_body_pair(ma0,maj), two_body_pair(mb0,mbj));
    out.nacta = nacta;
    out.nactb = nactb;
    return out;
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

/** \brief Read one same-spin II replacement entry for a determinant minor.
    \tparam Tc Matrix element type.
    \param slot Resolved same-spin two-electron intermediate slot.
    \param rows Row indices in the active orbital basis.
    \param cols Column indices in the active orbital basis.
    \param row_rm Removed determinant row.
    \param col_rm Removed determinant column.
    \param r_minor Row index in the minor matrix.
    \param k_minor Column index in the minor matrix.
    \param r_fixed Fixed replacement row.
    \param c_fixed Fixed replacement column.
    \return Same-spin two-electron replacement entry.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc same_ii_replacement(
    const same_ii_slot<Tc> &slot,
    const arma::uvec &rows,
    const arma::uvec &cols,
    const size_t row_rm,
    const size_t col_rm,
    const size_t r_minor,
    const size_t k_minor,
    const size_t r_fixed,
    const size_t c_fixed)
{
    const size_t r_full = minor_to_full(r_minor, row_rm);
    const size_t k_full = minor_to_full(k_minor, col_rm);
    return slot.get(r_fixed, c_fixed, rows(r_full), cols(k_full));
}

/** \brief Build a one-by-one minor from a two-by-two matrix.
    \tparam Tc Matrix element type.
    \param out Output minor matrix. Must have capacity at least 1-by-1.
    \param m Input matrix.
    \param r_rm Removed row.
    \param c_rm Removed column.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void minor_l2(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &m,
    const size_t r_rm,
    const size_t c_rm)
{
    const size_t r = 1 ^ r_rm;
    const size_t c = 1 ^ c_rm;
    out(0,0) = m(r,c);
}

/** \brief Build a two-by-two minor from a three-by-three matrix.
    \tparam Tc Matrix element type.
    \param out Output minor matrix. Must have capacity at least 2-by-2.
    \param m Input matrix.
    \param r_rm Removed row.
    \param c_rm Removed column.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void minor_l3(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &m,
    const size_t r_rm,
    const size_t c_rm)
{
    size_t ii = 0;
    for(size_t i=0; i<3; i++)
    {
        if(i == r_rm) continue;

        size_t jj = 0;
        for(size_t j=0; j<3; j++)
        {
            if(j == c_rm) continue;
            out(ii,jj) = m(i,j);
            jj++;
        }

        ii++;
    }
}

/** \brief Build a three-by-three minor from a four-by-four matrix.
    \tparam Tc Matrix element type.
    \param out Output minor matrix. Must have capacity at least 3-by-3.
    \param m Input matrix.
    \param r_rm Removed row.
    \param c_rm Removed column.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void minor_l4(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &m,
    const size_t r_rm,
    const size_t c_rm)
{
    size_t ii = 0;
    for(size_t i=0; i<4; i++)
    {
        if(i == r_rm) continue;

        size_t jj = 0;
        for(size_t j=0; j<4; j++)
        {
            if(j == c_rm) continue;
            out(ii,jj) = m(i,j);
            jj++;
        }

        ii++;
    }
}

/** \brief Build a determinant minor using specialised small-rank kernels.
    \tparam Tc Matrix element type.
    \param out Output minor matrix. Must have capacity at least n-1 by n-1.
    \param m Input matrix.
    \param n Active matrix rank.
    \param r_rm Removed row.
    \param c_rm Removed column.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void minor_into(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &m,
    const size_t n,
    const size_t r_rm,
    const size_t c_rm)
{
    if(n <= 1) return;
    if(n == 2) { minor_l2(out, m, r_rm, c_rm); return; }
    if(n == 3) { minor_l3(out, m, r_rm, c_rm); return; }
    if(n == 4) { minor_l4(out, m, r_rm, c_rm); return; }

    size_t ii = 0;
    for(size_t i=0; i<n; i++)
    {
        if(i == r_rm) continue;

        size_t jj = 0;
        for(size_t j=0; j<n; j++)
        {
            if(j == c_rm) continue;
            out(ii,jj) = m(i,j);
            jj++;
        }

        ii++;
    }
}

/** \brief Build an active determinant minor, compute its adjugate transpose, and pass it to a callback.
    \tparam Tc Matrix element type.
    \tparam Fn Callback type.
    \param D Input determinant matrix.
    \param l Active determinant rank.
    \param row_rm Removed row.
    \param col_rm Removed column.
    \param Dminor Scratch matrix for the minor. Must have capacity at least l-1 by l-1.
    \param cof_minor Scratch matrix for the adjugate transpose. Must have capacity at least l-1 by l-1.
    \param f Callback receiving minor matrix, adjugate transpose, and determinant.
    \ingroup gnme_wick
 **/
template<typename Tc, typename Fn>
inline void minor_adjt(
    const arma::Mat<Tc> &D,
    const size_t l,
    const size_t row_rm,
    const size_t col_rm,
    arma::Mat<Tc> &Dminor,
    arma::Mat<Tc> &cof_minor,
    Fn &&f)
{
    const size_t lm1 = l - 1;

    minor_into(Dminor, D, l, row_rm, col_rm);

    const Tc det_minor = adjugate_transpose(Dminor, lm1, cof_minor);
    f(Dminor, cof_minor, det_minor);
}

/** \brief Build a mixed determinant minor and return its determinant.
    \tparam Tc Matrix element type.
    \param D Branch-zero determinant matrix.
    \param Db Branch-one determinant matrix.
    \param bits Branch selector bits.
    \param offset Bit offset for determinant columns.
    \param l Active determinant rank.
    \param row_rm Removed row.
    \param col_rm Removed column.
    \param Dminor Scratch matrix for the mixed minor. Must have capacity at least l-1 by l-1.
    \return Determinant of the mixed minor.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc mixed_minor_det(
    const arma::Mat<Tc> &D,
    const arma::Mat<Tc> &Db,
    const uint64_t bits,
    const size_t offset,
    const size_t l,
    const size_t row_rm,
    const size_t col_rm,
    arma::Mat<Tc> &Dminor)
{
    const size_t lm1 = l - 1;

    for(size_t j=0; j<lm1; j++)
    {
        const size_t fj = minor_to_full(j, col_rm);
        const arma::Mat<Tc> &src = bit(bits, fj + offset) ? Db : D;

        for(size_t i=0; i<lm1; i++)
        {
            const size_t fi = minor_to_full(i, row_rm);
            Dminor(i,j) = src(fi,fj);
        }
    }

    return det(Dminor, lm1);
}

/** \brief Build a mixed determinant minor using the full matrix rank.
    \tparam Tc Matrix element type.
    \param D Branch-zero determinant matrix.
    \param Db Branch-one determinant matrix.
    \param bits Branch selector bits.
    \param offset Bit offset for determinant columns.
    \param row_rm Removed row.
    \param col_rm Removed column.
    \param Dminor Scratch matrix for the mixed minor.
    \return Determinant of the mixed minor.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc mixed_minor_det(
    const arma::Mat<Tc> &D,
    const arma::Mat<Tc> &Db,
    const uint64_t bits,
    const size_t offset,
    const size_t row_rm,
    const size_t col_rm,
    arma::Mat<Tc> &Dminor)
{
    return mixed_minor_det(D, Db, bits, offset, D.n_rows, row_rm, col_rm, Dminor);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_HELPERS_H
