#ifndef LIBGNME_WICK_EVAL_TWO_BODY_SAME_H
#define LIBGNME_WICK_EVAL_TWO_BODY_SAME_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "prepare.h"
#include "scratch.h"
#include "prepare_scratch.h"

namespace libgnme {
namespace wick_eval {

/** \brief Read one same-spin two-electron replacement entry.
    \tparam Tc Matrix element type.
    \param II Same-spin two-electron intermediate field.
    \param nact Total active dimension.
    \param mi First branch selector.
    \param mj Second branch selector.
    \param mk Third branch selector.
    \param ml Fourth branch selector.
    \param r0 Fixed row index.
    \param c0 Fixed column index.
    \param r1 Replacement row index.
    \param c1 Replacement column index.
    \return Two-electron replacement entry.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc two_body_same_ii(
    const arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    const size_t mi, const size_t mj,
    const size_t mk, const size_t ml,
    const size_t r0, const size_t c0,
    const size_t r1, const size_t c1)
{
    size_t p = two_body_pair(mi, mj);
    size_t q = two_body_pair(mk, ml);

    bool transpose;
    two_body_same_canonical(p, q, transpose);

    const arma::Mat<Tc> &M = II(p,q);

    const size_t col = nact * r0 + c0;
    const size_t row = c1 + nact * r1;

    return transpose ? M(col,row) : M(row,col);
}

/** \brief Read one same-spin II replacement entry for a determinant minor.
    \tparam Tc Matrix element type.
    \param II Same-spin two-electron intermediate field.
    \param nact Total active dimension.
    \param mi First branch selector.
    \param mj Second branch selector.
    \param mk Third branch selector.
    \param ml Fourth branch selector.
    \param rows Full determinant row labels.
    \param cols Full determinant column labels.
    \param row_rm Removed row in the full determinant.
    \param col_rm Removed column in the full determinant.
    \param r_minor Row index in the minor.
    \param k_minor Column index in the minor.
    \param r_fixed Fixed first row label.
    \param c_fixed Fixed first column label.
    \return Replacement entry.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc two_body_same_ii_replacement(
    const arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    const size_t mi, const size_t mj,
    const size_t mk, const size_t ml,
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

    return two_body_same_ii(
        II, nact,
        mi, mj, mk, ml,
        r_fixed, c_fixed,
        rows(r_full), cols(k_full));
}

/** \brief Evaluate same-spin two-body matrix element for zero overlap zeros and no excitations.
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param V0 Zeroth-order same-spin two-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_m0_l0(Tc &V, const arma::Col<Tc> &V0)
{
    V = V0(0);
}

/** \brief Evaluate same-spin two-body matrix element for nz = 0 using prepared scratch.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nact Total active dimension for flattened two-electron intermediates.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_m0(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    same_scratch<Tc> &work)
{
    const size_t nex = rows.n_elem;

    if(nex == 0)
    {
        two_body_same_m0_l0(V, V0);
        return;
    }

    if(nex == 1)
    {
        const size_t r0 = rows(0);
        const size_t c0 = cols(0);

        V = V0(0) * work.det0(0,0)
          - Tc(2.0) * XVX(0,0,0)(r0,c0);

        return;
    }

    if(nex == 2)
    {
        const Tc a00 = work.det0(0,0);
        const Tc a01 = work.det0(0,1);
        const Tc a10 = work.det0(1,0);
        const Tc a11 = work.det0(1,1);
        const Tc detD = a00 * a11 - a01 * a10;

        const arma::Mat<Tc> &JK = XVX(0,0,0);

        const size_t r0 = rows(0);
        const size_t r1 = rows(1);
        const size_t c0 = cols(0);
        const size_t c1 = cols(1);

        const Tc u0 = JK(r0,c0);
        const Tc u1 = JK(r1,c0);
        const Tc v0 = JK(r0,c1);
        const Tc v1 = JK(r1,c1);

        const Tc det_c0 = u0 * a11 - a01 * u1;
        const Tc det_c1 = a00 * v1 - v0 * a10;

        const Tc iiterm =
              two_body_same_ii(II, nact, 0, 0, 0, 0, r0, c0, r1, c1)
            - two_body_same_ii(II, nact, 0, 0, 0, 0, r0, c1, r1, c0)
            - two_body_same_ii(II, nact, 0, 0, 0, 0, r1, c0, r0, c1)
            + two_body_same_ii(II, nact, 0, 0, 0, 0, r1, c1, r0, c0);

        V = V0(0) * detD - Tc(2.0) * (det_c0 + det_c1) + Tc(0.5) * iiterm;
        return;
    }

    arma::Mat<Tc> &cofD = work.adjt_det;
    const Tc detD = adjugate_transpose(work.det0, cofD);

    V = V0(0) * detD;

    const arma::Mat<Tc> &JK = XVX(0,0,0);

    for(size_t k=0; k<nex; k++)
    {
        const size_t ck = cols(k);

        const Tc corr = column_replacement_correction(
            work.det0, cofD, k,
            [&](const size_t r) {
                return JK(rows(r), ck);
            });

        V -= Tc(2.0) * (detD + corr);
    }

    for(size_t i=0; i<nex; i++)
    for(size_t j=0; j<nex; j++)
    {
        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;
        const size_t r_fixed = rows(i);
        const size_t c_fixed = cols(j);

        minor_adjt(work.det0, i, j, work.det_mix2, work.adjt_det2,
            [&](const size_t lm1,
                const arma::Mat<Tc> &det_minor,
                const arma::Mat<Tc> &cof,
                const Tc det_minor_val)
            {
                for(size_t k=0; k<lm1; k++)
                {
                    const Tc corr = column_replacement_correction(
                        det_minor, cof, k,
                        [&](const size_t r) {
                            return two_body_same_ii_replacement(
                                II, nact,
                                0, 0, 0, 0,
                                rows, cols,
                                i, j, r, k,
                                r_fixed, c_fixed);
                        });

                    V += Tc(0.5 * phase) * (det_minor_val + corr);
                }
            });
    }
}

/** \brief Evaluate same-spin two-body matrix element for the generic nz > 0 case using prepared scratch.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nact Total active dimension for flattened two-electron intermediates.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const size_t &nz,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    same_scratch<Tc> &work)
{
    const size_t nex = rows.n_elem;

    if(nex == 0)
    {
        for_each_m_combination(2, nz, [&](uint64_t bits) {
            const size_t m0 = bit(bits, 0);
            const size_t m1 = bit(bits, 1);

            V += V0(m0 + m1);
        });

        return;
    }

    if(nex == 1)
    {
        const size_t r0 = rows(0);
        const size_t c0 = cols(0);

        for_each_m_combination(3, nz, [&](uint64_t bits) {
            const size_t m0 = bit(bits, 0);
            const size_t m1 = bit(bits, 1);
            const size_t m2 = bit(bits, 2);

            V += V0(m0 + m1) * (m2 ? work.det1(0,0) : work.det0(0,0))
               - Tc(2.0) * XVX(m0,m1,m2)(r0,c0);
        });

        return;
    }

    mix_dets_same(nex, nz, 2, work, [&](const uint64_t bits) {
        const size_t m0 = bit(bits, 0);
        const size_t m1 = bit(bits, 1);

        const Tc detDtmp = adjugate_transpose(work.det_mix, work.adjt_det);

        Tc contrib = V0(m0 + m1) * detDtmp;

        for(size_t k=0; k<nex; k++)
        {
            const size_t mk = bit(bits, k+2);
            const size_t ck = cols(k);

            const Tc corr = column_replacement_correction(
                work.det_mix, work.adjt_det, k,
                [&](const size_t r) {
                    return XVX(m0,m1,mk)(rows(r),ck);
                });

            contrib -= Tc(2.0) * (detDtmp + corr);
        }

        for(size_t i=0; i<nex; i++)
        for(size_t j=0; j<nex; j++)
        {
            const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;

            const size_t ri_fixed = rows(i);
            const size_t cj_fixed = cols(j);
            const size_t mj = bit(bits, j+2);

            minor_adjt(work.det_mix, i, j, work.det_mix2, work.adjt_det2,
                [&](const size_t lm1,
                    const arma::Mat<Tc> &det_minor,
                    const arma::Mat<Tc> &cof,
                    const Tc det_minor_val)
                {
                    for(size_t k2=0; k2<lm1; k2++)
                    {
                        const size_t k_full = minor_to_full(k2, j);
                        const size_t mk = bit(bits, k_full+2);

                        const Tc corr = column_replacement_correction(
                            det_minor, cof, k2,
                            [&](const size_t r) {
                                return two_body_same_ii_replacement(
                                    II, nact,
                                    m0, m1, mk, mj,
                                    rows, cols,
                                    i, j, r, k2,
                                    ri_fixed, cj_fixed);
                            });

                        contrib += Tc(0.5 * phase) * (det_minor_val + corr);
                    }
                });
        }

        V += contrib;
    });
}

/** \brief Evaluate same-spin two-body contribution using Wick contractions.
    \tparam Tc Matrix element type.
    \param xhp Particle-hole indices for bra state.
    \param whp Particle-hole indices for ket state.
    \param V Output two-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nactx Active dimension of bra-side reference.
    \param nactw Active dimension of ket-side reference.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same(
    arma::umat xhp, arma::umat whp,
    Tc &V,
    const size_t &nz,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nactx,
    const size_t nactw)
{
    V = Tc(0.0);

    const size_t nx = xhp.n_rows;
    const size_t nw = whp.n_rows;
    const size_t nex = nx + nw;

    if(nz > nex + 2) return;

    scratch<Tc> &store = local_scratch<Tc>();
    same_scratch<Tc> &work = store.aa;

    prepare_same(xhp, whp, nactx, nz, X, Y, work);

    const arma::uvec &rows = work.rows;
    const arma::uvec &cols = work.cols;
    const size_t nact = nactx + nactw;

    if(nz == 0)
    {
        two_body_same_m0(rows, cols, V, V0, XVX, II, nact, work);
        return;
    }

    two_body_same_gen(rows, cols, V, nz, V0, XVX, II, nact, work);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_TWO_BODY_SAME_H
