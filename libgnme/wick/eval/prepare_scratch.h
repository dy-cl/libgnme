#ifndef LIBGNME_WICK_EVAL_PREPARE_SCRATCH_H
#define LIBGNME_WICK_EVAL_PREPARE_SCRATCH_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "scratch.h"

namespace libgnme {
namespace wick_eval {

/** \brief Prepare one bra excitation index.
    \param xhp Bra particle-hole indices.
    \param i Bra excitation index.
    \param p Output position.
    \param rows Output row indices.
    \param cols Output column indices.
    \ingroup gnme_wick
 **/
inline void prepare_index_x(
    const arma::umat &xhp,
    const size_t i,
    const size_t p,
    arma::uvec &rows,
    arma::uvec &cols)
{
    rows(p) = xhp(i,1);
    cols(p) = xhp(i,0);
}

/** \brief Prepare one ket excitation index.
    \param whp Ket particle-hole indices.
    \param i Ket excitation index.
    \param p Output position.
    \param wshift Ket orbital index shift.
    \param rows Output row indices.
    \param cols Output column indices.
    \ingroup gnme_wick
 **/
inline void prepare_index_w(
    const arma::umat &whp,
    const size_t i,
    const size_t p,
    const size_t wshift,
    arma::uvec &rows,
    arma::uvec &cols)
{
    rows(p) = whp(i,0) + wshift;
    cols(p) = whp(i,1) + wshift;
}

/** \brief Prepare determinant indices for ranks up to four directly.
    \param xhp Bra particle-hole indices.
    \param whp Ket particle-hole indices.
    \param wshift Ket orbital index shift.
    \param rows Output row indices.
    \param cols Output column indices.
    \return True if the indices were prepared directly.
    \ingroup gnme_wick
 **/
inline bool prepare_indices_small(
    const arma::umat &xhp,
    const arma::umat &whp,
    const size_t wshift,
    arma::uvec &rows,
    arma::uvec &cols)
{
    const size_t nx = xhp.n_rows;
    const size_t nw = whp.n_rows;
    const size_t l = nx + nw;

    if(l > 4) return false;

    rows.set_size(l);
    cols.set_size(l);

    if(l == 0) return true;

    if(nx > 0) prepare_index_x(xhp, 0, 0, rows, cols);
    if(nx > 1) prepare_index_x(xhp, 1, 1, rows, cols);
    if(nx > 2) prepare_index_x(xhp, 2, 2, rows, cols);
    if(nx > 3) prepare_index_x(xhp, 3, 3, rows, cols);

    if(nw > 0) prepare_index_w(whp, 0, nx + 0, wshift, rows, cols);
    if(nw > 1) prepare_index_w(whp, 1, nx + 1, wshift, rows, cols);
    if(nw > 2) prepare_index_w(whp, 2, nx + 2, wshift, rows, cols);
    if(nw > 3) prepare_index_w(whp, 3, nx + 3, wshift, rows, cols);

    return true;
}

/** \brief Prepare determinant indices into scratch storage.
    \param xhp Bra particle-hole indices.
    \param whp Ket particle-hole indices.
    \param wshift Ket orbital index shift.
    \param rows Output row indices.
    \param cols Output column indices.
    \ingroup gnme_wick
 **/
inline void prepare_indices(
    const arma::umat &xhp,
    const arma::umat &whp,
    const size_t wshift,
    arma::uvec &rows,
    arma::uvec &cols)
{
    if(prepare_indices_small(xhp, whp, wshift, rows, cols))
        return;

    const size_t nx = xhp.n_rows;
    const size_t nw = whp.n_rows;
    const size_t l = nx + nw;

    rows.set_size(l);
    cols.set_size(l);

    size_t p = 0;
    for(size_t i=0; i<nx; i++)
    {
        rows(p) = xhp(i,1);
        cols(p) = xhp(i,0);
        p++;
    }

    for(size_t i=0; i<nw; i++)
    {
        rows(p) = whp(i,0) + wshift;
        cols(p) = whp(i,1) + wshift;
        p++;
    }
}

/** \brief Prepare rank-1 same-spin determinant branch directly.
    \tparam Tc Matrix element type.
    \param X Lower-triangular contraction matrix.
    \param rows Row indices.
    \param cols Column indices.
    \param D Output determinant.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same_l1(
    const arma::Mat<Tc> &X,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<Tc> &D)
{
    D.set_size(1,1);
    D(0,0) = X(rows(0),cols(0));
}

/** \brief Prepare rank-2 same-spin determinant branch directly.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same_l2(
    const arma::Mat<Tc> &X,
    const arma::Mat<Tc> &Y,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<Tc> &D)
{
    D.set_size(2,2);

    const size_t r0 = rows(0);
    const size_t r1 = rows(1);
    const size_t c0 = cols(0);
    const size_t c1 = cols(1);

    D(0,0) = X(r0,c0);
    D(0,1) = Y(r0,c1);
    D(1,0) = X(r1,c0);
    D(1,1) = X(r1,c1);
}

/** \brief Prepare rank-3 same-spin determinant branch directly.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same_l3(
    const arma::Mat<Tc> &X,
    const arma::Mat<Tc> &Y,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<Tc> &D)
{
    D.set_size(3,3);

    const size_t r0 = rows(0);
    const size_t r1 = rows(1);
    const size_t r2 = rows(2);

    const size_t c0 = cols(0);
    const size_t c1 = cols(1);
    const size_t c2 = cols(2);

    D(0,0) = X(r0,c0);
    D(0,1) = Y(r0,c1);
    D(0,2) = Y(r0,c2);

    D(1,0) = X(r1,c0);
    D(1,1) = X(r1,c1);
    D(1,2) = Y(r1,c2);

    D(2,0) = X(r2,c0);
    D(2,1) = X(r2,c1);
    D(2,2) = X(r2,c2);
}

/** \brief Prepare rank-4 same-spin determinant branch directly.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same_l4(
    const arma::Mat<Tc> &X,
    const arma::Mat<Tc> &Y,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<Tc> &D)
{
    D.set_size(4,4);

    const size_t r0 = rows(0);
    const size_t r1 = rows(1);
    const size_t r2 = rows(2);
    const size_t r3 = rows(3);

    const size_t c0 = cols(0);
    const size_t c1 = cols(1);
    const size_t c2 = cols(2);
    const size_t c3 = cols(3);

    D(0,0) = X(r0,c0);
    D(0,1) = Y(r0,c1);
    D(0,2) = Y(r0,c2);
    D(0,3) = Y(r0,c3);

    D(1,0) = X(r1,c0);
    D(1,1) = X(r1,c1);
    D(1,2) = Y(r1,c2);
    D(1,3) = Y(r1,c3);

    D(2,0) = X(r2,c0);
    D(2,1) = X(r2,c1);
    D(2,2) = X(r2,c2);
    D(2,3) = Y(r2,c3);

    D(3,0) = X(r3,c0);
    D(3,1) = X(r3,c1);
    D(3,2) = X(r3,c2);
    D(3,3) = X(r3,c3);
}

/** \brief Prepare one same-spin determinant branch.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same_branch(
    const arma::Mat<Tc> &X,
    const arma::Mat<Tc> &Y,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<Tc> &D)
{
    const size_t l = rows.n_elem;

    if(l == 0)
    {
        D.set_size(0,0);
        return;
    }

    if(l == 1)
    {
        prepare_same_l1(X, rows, cols, D);
        return;
    }

    if(l == 2)
    {
        prepare_same_l2(X, Y, rows, cols, D);
        return;
    }

    if(l == 3)
    {
        prepare_same_l3(X, Y, rows, cols, D);
        return;
    }

    if(l == 4)
    {
        prepare_same_l4(X, Y, rows, cols, D);
        return;
    }

    build_det(X, Y, rows, cols, D);
}

/** \brief Prepare same-spin scratch determinant branches.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same(
    const arma::umat &xhp,
    const arma::umat &whp,
    const size_t wshift,
    const size_t nz,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    same_scratch<Tc> &work)
{
    const size_t l = xhp.n_rows + whp.n_rows;

    work.ensure(l);
    prepare_indices(xhp, whp, wshift, work.rows, work.cols);

    prepare_same_branch(X(0), Y(0), work.rows, work.cols, work.det0);

    if(nz != 0)
        prepare_same_branch(X(1), Y(1), work.rows, work.cols, work.det1);
}

/** \brief Copy one column from source to destination.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void copy_column(
    const arma::Mat<Tc> &src,
    arma::Mat<Tc> &dst,
    const size_t col)
{
    const size_t n = src.n_rows;

    for(size_t r=0; r<n; r++)
        dst(r,col) = src(r,col);
}

/** \brief Form mixed same-spin determinants incrementally.
    \tparam Tc Matrix element type.
    \tparam Fn Callback type.
    \param l Excitation rank.
    \param nz Number of zero-overlap orbital pairs.
    \param pbits Number of leading operator selector bits.
    \param work Same-spin scratch.
    \param f Callback receiving bitstring.
    \ingroup gnme_wick
 **/
template<typename Tc, typename Fn>
inline void mix_dets_same(
    const size_t l,
    const size_t nz,
    const size_t pbits,
    same_scratch<Tc> &work,
    Fn &&f)
{
    bool first = true;
    uint64_t prev_cbits = 0;

    for_each_m_combination(l + pbits, nz, [&](const uint64_t bits) {
        const uint64_t cbits = bits >> pbits;

        if(first)
        {
            work.det_mix.set_size(l,l);

            for(size_t c=0; c<l; c++)
            {
                if(bit(cbits,c))
                    copy_column(work.det1, work.det_mix, c);
                else
                    copy_column(work.det0, work.det_mix, c);
            }

            first = false;
        }
        else
        {
            uint64_t changed = prev_cbits ^ cbits;

            while(changed != 0)
            {
                const size_t c = __builtin_ctzll(changed);

                if(bit(cbits,c))
                    copy_column(work.det1, work.det_mix, c);
                else
                    copy_column(work.det0, work.det_mix, c);

                changed &= changed - 1;
            }
        }

        prev_cbits = cbits;
        f(bits);
    });
}

/** \brief Form mixed alpha determinant incrementally.
    \tparam Tc Matrix element type.
    \param bits Zero-distribution bitstring.
    \param offset Bit offset for determinant columns.
    \param same Prepared same-spin scratch storage.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_deta(
    const uint64_t bits,
    const size_t offset,
    same_scratch<Tc> &same,
    diff_scratch<Tc> &work)
{
    const size_t l = same.rows.n_elem;
    work.deta_mix.set_size(l,l);

    for(size_t c=0; c<l; c++)
    {
        if(bit(bits,c+offset))
            copy_column(same.det1, work.deta_mix, c);
        else
            copy_column(same.det0, work.deta_mix, c);
    }
}


/** \brief Form mixed beta determinant incrementally.
    \tparam Tc Matrix element type.
    \param bits Zero-distribution bitstring.
    \param offset Bit offset for determinant columns.
    \param same Prepared same-spin scratch storage.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_detb(
    const uint64_t bits,
    const size_t offset,
    same_scratch<Tc> &same,
    diff_scratch<Tc> &work)
{
    const size_t l = same.rows.n_elem;
    work.detb_mix.set_size(l,l);

    for(size_t c=0; c<l; c++)
    {
        if(bit(bits,c+offset))
            copy_column(same.det1, work.detb_mix, c);
        else
            copy_column(same.det0, work.detb_mix, c);
    }
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_PREPARE_SCRATCH_H
