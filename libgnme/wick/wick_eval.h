#ifndef LIBGNME_WICK_EVAL_H
#define LIBGNME_WICK_EVAL_H 

#include <armadillo>
#include <cassert>
#include <vector>

namespace libgnme {
namespace wick_eval {
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

    /** \brief Build same-spin determinant rows and columns from particle-hole lists.
    \param xhp Particle-hole indices for bra state.
    \param whp Particle-hole indices for ket state.
    \param rows Output row indices.
    \param cols Output column indices.
    \ingroup gnme_wick
     **/
    inline void indices(
        const arma::umat &xhp, const arma::umat &whp,
        arma::uvec &rows, arma::uvec &cols)
    {
        const size_t nx = xhp.n_rows;
        const size_t nw = whp.n_rows;

        if(nx == 0 && nw == 0)
        {
            rows.set_size(0);
            cols.set_size(0);
        }
        else if(nx == 0)
        {
            rows = whp.col(0);
            cols = whp.col(1);
        }
        else if(nw == 0)
        {
            rows = xhp.col(1);
            cols = xhp.col(0);
        }
        else
        {
            rows = arma::join_cols(xhp.col(1), whp.col(0));
            cols = arma::join_cols(xhp.col(0), whp.col(1));
        }
    }

    /** \brief Build a 1x1 Wick determinant from lower and upper contraction matrices.
    \tparam T Matrix element type.
    \param X Lower-triangular contraction matrix.
    \param Y Upper-triangular contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param D Output determinant matrix.
    \ingroup gnme_wick
     **/
    template<typename T>
    inline void build_det1(
        const arma::Mat<T> &X, const arma::Mat<T> &Y,
        const arma::uvec &rows, const arma::uvec &cols,
        arma::Mat<T> &D)
    {
        (void)Y;
        D.set_size(1,1);
        D(0,0) = X(rows(0),cols(0));
    }

    /** \brief Build a 2x2 Wick determinant from lower and upper contraction matrices.
    \tparam T Matrix element type.
    \param X Lower-triangular contraction matrix.
    \param Y Upper-triangular contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param D Output determinant matrix.
    \ingroup gnme_wick
     **/
    template<typename T>
    inline void build_det2(
        const arma::Mat<T> &X, const arma::Mat<T> &Y,
        const arma::uvec &rows, const arma::uvec &cols,
        arma::Mat<T> &D)
    {
        D.set_size(2,2);

        const size_t r0 = rows(0), r1 = rows(1);
        const size_t c0 = cols(0), c1 = cols(1);

        D(0,0) = X(r0,c0); D(0,1) = Y(r0,c1);
        D(1,0) = X(r1,c0); D(1,1) = X(r1,c1);
    }

    /** \brief Build a 3x3 Wick determinant from lower and upper contraction matrices.
    \tparam T Matrix element type.
    \param X Lower-triangular contraction matrix.
    \param Y Upper-triangular contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param D Output determinant matrix.
    \ingroup gnme_wick
     **/
    template<typename T>
    inline void build_det3(
        const arma::Mat<T> &X, const arma::Mat<T> &Y,
        const arma::uvec &rows, const arma::uvec &cols,
        arma::Mat<T> &D)
    {
        D.set_size(3,3);

        const size_t r0 = rows(0), r1 = rows(1), r2 = rows(2);
        const size_t c0 = cols(0), c1 = cols(1), c2 = cols(2);

        D(0,0) = X(r0,c0); D(0,1) = Y(r0,c1); D(0,2) = Y(r0,c2);
        D(1,0) = X(r1,c0); D(1,1) = X(r1,c1); D(1,2) = Y(r1,c2);
        D(2,0) = X(r2,c0); D(2,1) = X(r2,c1); D(2,2) = X(r2,c2);
    }

    /** \brief Build a 4x4 Wick determinant from lower and upper contraction matrices.
    \tparam T Matrix element type.
    \param X Lower-triangular contraction matrix.
    \param Y Upper-triangular contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param D Output determinant matrix.
    \ingroup gnme_wick
     **/
    template<typename T>
    inline void build_det4(
        const arma::Mat<T> &X, const arma::Mat<T> &Y,
        const arma::uvec &rows, const arma::uvec &cols,
        arma::Mat<T> &D)
    {
        D.set_size(4,4);

        const size_t r0 = rows(0), r1 = rows(1), r2 = rows(2), r3 = rows(3);
        const size_t c0 = cols(0), c1 = cols(1), c2 = cols(2), c3 = cols(3);

        D(0,0) = X(r0,c0); D(0,1) = Y(r0,c1); D(0,2) = Y(r0,c2); D(0,3) = Y(r0,c3);
        D(1,0) = X(r1,c0); D(1,1) = X(r1,c1); D(1,2) = Y(r1,c2); D(1,3) = Y(r1,c3);
        D(2,0) = X(r2,c0); D(2,1) = X(r2,c1); D(2,2) = X(r2,c2); D(2,3) = Y(r2,c3);
        D(3,0) = X(r3,c0); D(3,1) = X(r3,c1); D(3,2) = X(r3,c2); D(3,3) = X(r3,c3);
    }

    /** \brief Build a Wick determinant from lower and upper contraction matrices.
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
        const arma::Mat<T> &X, const arma::Mat<T> &Y,
        const arma::uvec &rows, const arma::uvec &cols,
        arma::Mat<T> &D)
    {
        const size_t n = rows.n_elem;

        if(n == 1) { build_det1(X, Y, rows, cols, D); return; }
        if(n == 2) { build_det2(X, Y, rows, cols, D); return; }
        if(n == 3) { build_det3(X, Y, rows, cols, D); return; }
        if(n == 4) { build_det4(X, Y, rows, cols, D); return; }

        D.set_size(n,n);

        for(size_t j=0; j<n; j++)
        for(size_t i=0; i<n; i++)
            D(i,j) = (i >= j) ? X(rows(i),cols(j)) : Y(rows(i),cols(j));
    }

    /** \brief Build a Wick matrix from one contraction matrix.
    \tparam T Matrix element type.
    \param A Contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param D Output determinant matrix.
    \ingroup gnme_wick
     **/
    template<typename T>
    inline void build_mat(
        const arma::Mat<T> &A,
        const arma::uvec &rows, const arma::uvec &cols,
        arma::Mat<T> &D)
    {
        const size_t n = rows.n_elem;
        D.set_size(n,n);

        for(size_t j=0; j<n; j++)
        for(size_t i=0; i<n; i++)
            D(i,j) = A(rows(i),cols(j));
    }

    /** \brief Replace selected columns of one determinant matrix by columns from another.
    \tparam T Matrix element type.
    \param D Non-zero-overlap branch determinant.
    \param Dbar Zero-overlap branch determinant.
    \param mask Column replacement mask.
    \param out Output mixed determinant.
    \ingroup gnme_wick
     **/
    template<typename T>
    inline void mix_det(
        const arma::Mat<T> &D, const arma::Mat<T> &Dbar,
        const std::vector<size_t> &mask, arma::Mat<T> &out)
    {
        const size_t n = D.n_rows;
        out.set_size(n,n);

        for(size_t j=0; j<n; j++)
        {
            const arma::Mat<T> &src = mask[j] ? Dbar : D;

            for(size_t i=0; i<n; i++)
                out(i,j) = src(i,j);
        }
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

#endif // LIBGNME_WICK_EVAL_H
