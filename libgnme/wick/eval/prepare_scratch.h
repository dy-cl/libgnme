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

/** \brief Build a one-by-one replacement determinant.
    \tparam Tc Matrix element type.
    \param X Lower-triangular replacement source matrix.
    \param rows Row indices in the active orbital basis.
    \param cols Column indices in the active orbital basis.
    \param D Output determinant scratch matrix. Must already have capacity at least 1-by-1.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same_l1(
    const arma::Mat<Tc> &X,
    const arma::uvec &rows,
    const arma::uvec &cols,
    arma::Mat<Tc> &D)
{
    D(0,0) = X(rows(0), cols(0));
}

/** \brief Build a two-by-two replacement determinant.
    \tparam Tc Matrix element type.
    \param X Lower-triangular replacement source matrix.
    \param Y Strictly upper-triangular replacement source matrix.
    \param rows Row indices in the active orbital basis.
    \param cols Column indices in the active orbital basis.
    \param D Output determinant scratch matrix. Must already have capacity at least 2-by-2.
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
    const size_t r0 = rows(0), r1 = rows(1);
    const size_t c0 = cols(0), c1 = cols(1);

    D(0,0) = X(r0,c0);
    D(0,1) = Y(r0,c1);
    D(1,0) = X(r1,c0);
    D(1,1) = X(r1,c1);
}

/** \brief Build a three-by-three replacement determinant.
    \tparam Tc Matrix element type.
    \param X Lower-triangular replacement source matrix.
    \param Y Strictly upper-triangular replacement source matrix.
    \param rows Row indices in the active orbital basis.
    \param cols Column indices in the active orbital basis.
    \param D Output determinant scratch matrix. Must already have capacity at least 3-by-3.
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
    const size_t r0 = rows(0), r1 = rows(1), r2 = rows(2);
    const size_t c0 = cols(0), c1 = cols(1), c2 = cols(2);

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

/** \brief Build a four-by-four replacement determinant.
    \tparam Tc Matrix element type.
    \param X Lower-triangular replacement source matrix.
    \param Y Strictly upper-triangular replacement source matrix.
    \param rows Row indices in the active orbital basis.
    \param cols Column indices in the active orbital basis.
    \param D Output determinant scratch matrix. Must already have capacity at least 4-by-4.
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
    const size_t r0 = rows(0), r1 = rows(1), r2 = rows(2), r3 = rows(3);
    const size_t c0 = cols(0), c1 = cols(1), c2 = cols(2), c3 = cols(3);

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

/** \brief Build a same-spin replacement determinant using specialised small-rank kernels.
    \tparam Tc Matrix element type.
    \param X Lower-triangular replacement source matrix.
    \param Y Strictly upper-triangular replacement source matrix.
    \param rows Row indices in the active orbital basis.
    \param cols Column indices in the active orbital basis.
    \param l Active determinant rank.
    \param D Output determinant scratch matrix. Must already have capacity at least l-by-l.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void prepare_same_branch(
    const arma::Mat<Tc> &X,
    const arma::Mat<Tc> &Y,
    const arma::uvec &rows,
    const arma::uvec &cols,
    const size_t l,
    arma::Mat<Tc> &D)
{
    if(l == 0) return;
    if(l == 1) { prepare_same_l1(X, rows, cols, D); return; }
    if(l == 2) { prepare_same_l2(X, Y, rows, cols, D); return; }
    if(l == 3) { prepare_same_l3(X, Y, rows, cols, D); return; }
    if(l == 4) { prepare_same_l4(X, Y, rows, cols, D); return; }

    for(size_t i=0; i<l; ++i) {
        const size_t r = rows(i);
        for(size_t j=0; j<=i; ++j) D(i,j) = X(r, cols(j));
        for(size_t j=i+1; j<l; ++j) D(i,j) = Y(r, cols(j));
    }
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
    
    work.ensure_capacity(l);
    work.set_active(l);

    prepare_indices(xhp, whp, wshift, work.rows, work.cols);
    prepare_same_branch(X(0), Y(0), work.rows, work.cols, l, work.det0);

    if(nz != 0)
        prepare_same_branch(X(1), Y(1), work.rows, work.cols, l, work.det1);
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

/** \brief Mix one column from two determinant matrices.
    \tparam Tc Matrix element type.
    \param out Output mixed determinant. Must already have capacity at least 1-by-1.
    \param det0 Determinant built from branch zero.
    \param det1 Determinant built from branch one.
    \param bits Column selector bits. Bit c selects column c from det1 when set, otherwise det0.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_columns_l1(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &det0,
    const arma::Mat<Tc> &det1,
    const uint64_t bits)
{
    out(0,0) = bit(bits, 0) ? det1(0,0) : det0(0,0);
}

/** \brief Mix two columns from two determinant matrices.
    \tparam Tc Matrix element type.
    \param out Output mixed determinant. Must already have capacity at least 2-by-2.
    \param det0 Determinant built from branch zero.
    \param det1 Determinant built from branch one.
    \param bits Column selector bits. Bit c selects column c from det1 when set, otherwise det0.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_columns_l2(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &det0,
    const arma::Mat<Tc> &det1,
    const uint64_t bits)
{
    const bool b0 = bit(bits, 0), b1 = bit(bits, 1);

    out(0,0) = b0 ? det1(0,0) : det0(0,0);
    out(1,0) = b0 ? det1(1,0) : det0(1,0);

    out(0,1) = b1 ? det1(0,1) : det0(0,1);
    out(1,1) = b1 ? det1(1,1) : det0(1,1);
}

/** \brief Mix three columns from two determinant matrices.
    \tparam Tc Matrix element type.
    \param out Output mixed determinant. Must already have capacity at least 3-by-3.
    \param det0 Determinant built from branch zero.
    \param det1 Determinant built from branch one.
    \param bits Column selector bits. Bit c selects column c from det1 when set, otherwise det0.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_columns_l3(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &det0,
    const arma::Mat<Tc> &det1,
    const uint64_t bits)
{
    const bool b0 = bit(bits, 0), b1 = bit(bits, 1), b2 = bit(bits, 2);

    out(0,0) = b0 ? det1(0,0) : det0(0,0);
    out(1,0) = b0 ? det1(1,0) : det0(1,0);
    out(2,0) = b0 ? det1(2,0) : det0(2,0);

    out(0,1) = b1 ? det1(0,1) : det0(0,1);
    out(1,1) = b1 ? det1(1,1) : det0(1,1);
    out(2,1) = b1 ? det1(2,1) : det0(2,1);

    out(0,2) = b2 ? det1(0,2) : det0(0,2);
    out(1,2) = b2 ? det1(1,2) : det0(1,2);
    out(2,2) = b2 ? det1(2,2) : det0(2,2);
}

/** \brief Mix four columns from two determinant matrices.
    \tparam Tc Matrix element type.
    \param out Output mixed determinant. Must already have capacity at least 4-by-4.
    \param det0 Determinant built from branch zero.
    \param det1 Determinant built from branch one.
    \param bits Column selector bits. Bit c selects column c from det1 when set, otherwise det0.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_columns_l4(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &det0,
    const arma::Mat<Tc> &det1,
    const uint64_t bits)
{
    for(size_t c=0; c<4; ++c) {
        const arma::Mat<Tc> &src = bit(bits, c) ? det1 : det0;
        out(0,c) = src(0,c);
        out(1,c) = src(1,c);
        out(2,c) = src(2,c);
        out(3,c) = src(3,c);
    }
}

/** \brief Mix columns from two determinant matrices using specialised small-rank kernels.
    \tparam Tc Matrix element type.
    \param out Output mixed determinant. Must already have capacity at least l-by-l.
    \param det0 Determinant built from branch zero.
    \param det1 Determinant built from branch one.
    \param l Active determinant rank.
    \param bits Column selector bits. Bit c selects column c from det1 when set, otherwise det0.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void mix_columns(
    arma::Mat<Tc> &out,
    const arma::Mat<Tc> &det0,
    const arma::Mat<Tc> &det1,
    const size_t l,
    const uint64_t bits)
{
    if(l == 0) return;
    if(l == 1) { mix_columns_l1(out, det0, det1, bits); return; }
    if(l == 2) { mix_columns_l2(out, det0, det1, bits); return; }
    if(l == 3) { mix_columns_l3(out, det0, det1, bits); return; }
    if(l == 4) { mix_columns_l4(out, det0, det1, bits); return; }

    for(size_t c=0; c<l; ++c) {
        const arma::Mat<Tc> &src = bit(bits, c) ? det1 : det0;
        for(size_t r=0; r<l; ++r) out(r,c) = src(r,c);
    }
}

/** \brief Iterate over mixed determinant branches for a same-spin zero-overlap expansion.
    \tparam Tc Matrix element type.
    \tparam Fn Callback type.
    \param l Active determinant rank.
    \param nz Number of zero-overlap orbital pairs.
    \param pbits Number of leading non-column selector bits in the branch encoding.
    \param work Same-spin scratch storage.
    \param f Callback called once for each branch bit pattern.
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
    for_each_m_combination(l + pbits, nz, [&](const uint64_t bits) {
        const uint64_t cbits = bits >> pbits;
        mix_columns(work.det_mix, work.det0, work.det1, l, cbits);
        f(bits);
    });
}

/** \brief Form mixed alpha determinant from prepared same-spin branch determinants.
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
    const size_t l = same.l;
    const uint64_t cbits = bits >> offset;
    mix_columns(work.deta_mix, same.det0, same.det1, l, cbits);
}

/** \brief Form mixed beta determinant from prepared same-spin branch determinants.
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
    const size_t l = same.l;
    const uint64_t cbits = bits >> offset;
    mix_columns(work.detb_mix, same.det0, same.det1, l, cbits);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_PREPARE_SCRATCH_H
