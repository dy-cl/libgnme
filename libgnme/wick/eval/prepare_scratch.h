#ifndef LIBGNME_WICK_EVAL_PREPARE_SCRATCH_H
#define LIBGNME_WICK_EVAL_PREPARE_SCRATCH_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "scratch.h"

namespace libgnme {
namespace wick_eval {

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

/** \brief Prepare different-spin scratch determinant branches.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_diff(
    const arma::umat &xahp,
    const arma::umat &wahp,
    const arma::umat &xbhp,
    const arma::umat &wbhp,
    const size_t nactxa,
    const size_t nactxb,
    const size_t nza,
    const size_t nzb,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Ya,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::field<arma::Mat<Tc> > &Yb,
    arma::uvec &rowa,
    arma::uvec &cola,
    arma::uvec &rowb,
    arma::uvec &colb,
    diff_scratch<Tc> &work)
{
    const size_t la = xahp.n_rows + wahp.n_rows;
    const size_t lb = xbhp.n_rows + wbhp.n_rows;

    work.ensure(la, lb);

    prepare_indices(xahp, wahp, nactxa, rowa, cola);
    prepare_indices(xbhp, wbhp, nactxb, rowb, colb);

    prepare_same_branch(Xa(0), Ya(0), rowa, cola, work.deta0);
    prepare_same_branch(Xb(0), Yb(0), rowb, colb, work.detb0);

    if(nza != 0)
        prepare_same_branch(Xa(1), Ya(1), rowa, cola, work.deta1);

    if(nzb != 0)
        prepare_same_branch(Xb(1), Yb(1), rowb, colb, work.detb1);
}

/** \brief Form mixed alpha determinant incrementally.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_deta(
    const uint64_t bits,
    const size_t offset,
    diff_scratch<Tc> &work)
{
    const size_t l = work.deta0.n_rows;
    work.deta_mix.set_size(l,l);

    for(size_t c=0; c<l; c++)
    {
        if(bit(bits,c+offset))
            copy_column(work.deta1, work.deta_mix, c);
        else
            copy_column(work.deta0, work.deta_mix, c);
    }
}

/** \brief Form mixed beta determinant incrementally.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_detb(
    const uint64_t bits,
    const size_t offset,
    diff_scratch<Tc> &work)
{
    const size_t l = work.detb0.n_rows;
    work.detb_mix.set_size(l,l);

    for(size_t c=0; c<l; c++)
    {
        if(bit(bits,c+offset))
            copy_column(work.detb1, work.detb_mix, c);
        else
            copy_column(work.detb0, work.detb_mix, c);
    }
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_PREPARE_SCRATCH_H
