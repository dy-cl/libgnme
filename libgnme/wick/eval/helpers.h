#ifndef LIBGNME_WICK_EVAL_HELPERS_H
#define LIBGNME_WICK_EVAL_HELPERS_H

#include <armadillo>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <vector>

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
