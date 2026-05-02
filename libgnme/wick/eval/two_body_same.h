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

/** \brief Evaluate same-spin two-body contribution using prepared Wick contractions.
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nact Total active dimension for flattened two-electron intermediates.
    \param work Prepared same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same(
    Tc &V,
    const size_t &nz,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    same_scratch<Tc> &work)
{
    V = Tc(0.0);

    const size_t nex = work.l;
    if(nz > nex + 2) return;

    const arma::uvec &rows = work.rows;
    const arma::uvec &cols = work.cols;

    if(nz == 0)
    {
        two_body_same_m0(rows, cols, V, V0, XVX, II, nact, work);
        return;
    }

    two_body_same_gen(rows, cols, V, nz, V0, XVX, II, nact, work);
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

/** \brief Evaluate same-spin two-body matrix element for zero overlap zeros and one excitation.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_m0_l1(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    same_scratch<Tc> &work)
{
    const size_t r0 = rows(0);
    const size_t c0 = cols(0);

    V = V0(0) * work.det0(0,0)
      - Tc(2.0) * XVX(0,0,0)(r0,c0);
}

/** \brief Evaluate same-spin two-body matrix element for zero overlap zeros and two excitations.
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
inline void two_body_same_m0_l2(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    same_scratch<Tc> &work)
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

    const same_ii_slot<Tc> ii0000 = resolve_same_ii_slot(II, nact, 0, 0, 0, 0);

    const Tc iiterm =
        ii0000.get(r0, c0, r1, c1)
      - ii0000.get(r0, c1, r1, c0)
      - ii0000.get(r1, c0, r0, c1)
      + ii0000.get(r1, c1, r0, c0);

    V = V0(0) * detD - Tc(2.0) * (det_c0 + det_c1) + Tc(0.5) * iiterm;
}


/** \brief Evaluate same-spin two-body matrix element for nz = 0 and three excitations.
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
inline void two_body_same_m0_l3(
    const arma::uvec &rows,
    const arma::uvec &cols,
    Tc &V,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    same_scratch<Tc> &work)
{
    arma::Mat<Tc> &cofD = work.adjt_det;
    const Tc detD = adjugate_transpose(work.det0, work.l, cofD);

    const arma::Mat<Tc> &JK = XVX(0,0,0);
    const same_ii_slot<Tc> ii0000 = resolve_same_ii_slot(II, nact, 0, 0, 0, 0);

    Tc vterm = Tc(0.0);
    for(size_t c=0; c<3; c++)
        for(size_t r=0; r<3; r++)
            vterm += cofD(r,c) * JK(rows(r), cols(c));

    Tc jterm = Tc(0.0);
    for(size_t i=0; i<3; i++)
    {
        const size_t ra0 = (i == 0) ? 1 : 0;
        const size_t ra1 = (i == 2) ? 1 : 2;
        const size_t ri = rows(i);

        for(size_t j=0; j<3; j++)
        {
            const size_t cb0 = (j == 0) ? 1 : 0;
            const size_t cb1 = (j == 2) ? 1 : 2;
            const size_t cj = cols(j);

            const Tc phase = ((i + j) % 2) ? Tc(-1.0) : Tc(1.0);

            const Tc m00 = work.det0(ra0, cb0);
            const Tc m01 = work.det0(ra0, cb1);
            const Tc m10 = work.det0(ra1, cb0);
            const Tc m11 = work.det0(ra1, cb1);
            
            const Tc j00 = ii0000.get(ri, cj, rows(ra0), cols(cb0));
            const Tc j01 = ii0000.get(ri, cj, rows(ra0), cols(cb1));
            const Tc j10 = ii0000.get(ri, cj, rows(ra1), cols(cb0));
            const Tc j11 = ii0000.get(ri, cj, rows(ra1), cols(cb1));

            jterm += phase * (m11 * j00 - m10 * j01 - m01 * j10 + m00 * j11);
        }
    }

    V = V0(0) * detD - Tc(2.0) * vterm + Tc(0.5) * jterm;
}

/** \brief Evaluate same-spin two-body matrix element for zero overlap zeros and general excitation rank.
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
inline void two_body_same_m0_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact,
    same_scratch<Tc> &work)
{
    const size_t nex = work.l;

    arma::Mat<Tc> &cofD = work.adjt_det;
    const Tc detD = adjugate_transpose(work.det0, work.l, cofD);

    V = V0(0) * detD;

    const arma::Mat<Tc> &JK = XVX(0,0,0);
    const same_ii_slot<Tc> ii0000 = resolve_same_ii_slot(II, nact, 0, 0, 0, 0);

    for(size_t k=0; k<nex; k++)
    {
        const size_t ck = cols(k);

        const Tc corr = column_replacement_correction(
            work.det0, cofD, work.l, k,
            [&](const size_t r) {
                return JK(rows(r), ck);
            });

        V -= Tc(2.0) * corr;
    }

    for(size_t i=0; i<nex; i++)
    for(size_t j=0; j<nex; j++)
    {
        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;
        const size_t r_fixed = rows(i);
        const size_t c_fixed = cols(j);

        minor_adjt(work.det0, work.l, i, j, work.det_mix2, work.adjt_det2,
        [&](const arma::Mat<Tc> &det_minor,
            const arma::Mat<Tc> &cof,
            const Tc det_minor_val)
        {
            const size_t lm1 = nex - 1;

            for(size_t k=0; k<lm1; k++)
            {
                const Tc corr = column_replacement_correction(
                    det_minor, cof, lm1, k,
                    [&](const size_t r) {
                        return same_ii_replacement(
                            ii0000,
                            rows, cols,
                            i, j,
                            r, k,
                            rows(i),
                            cols(j));
                    });

                V += Tc(0.5 * phase) * corr;
            }
        });
    }
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
    const size_t nex = work.l;

    if(nex == 0)
    {
        two_body_same_m0_l0(V, V0);
        return;
    }

    if(nex == 1)
    {
        two_body_same_m0_l1(rows, cols, V, V0, XVX, work);
        return;
    }

    if(nex == 2)
    {
        two_body_same_m0_l2(rows, cols, V, V0, XVX, II, nact, work);
        return;
    }

    if(nex == 3)
    {
        two_body_same_m0_l3(rows, cols, V, V0, XVX, II, nact, work);
        return;
    }

    two_body_same_m0_gen(rows, cols, V, V0, XVX, II, nact, work);
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
    const size_t nex = work.l;

    if(nex == 0)
    {
        for_each_m_combination(2, nz, [&](const uint64_t bits) {
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

        for_each_m_combination(3, nz, [&](const uint64_t bits) {
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

        const Tc detDtmp = adjugate_transpose(work.det_mix, work.l, work.adjt_det);

        Tc contrib = V0(m0 + m1) * detDtmp;

        for(size_t k=0; k<nex; k++)
        {
            const size_t mk = bit(bits, k + 2);
            const size_t ck = cols(k);

            const arma::Mat<Tc> &X = XVX(m0,m1,mk);

            const Tc corr = column_replacement_correction(
                work.det_mix, work.adjt_det, work.l, k,
                [&](const size_t r) {
                    return X(rows(r), ck);
                });

            contrib -= Tc(2.0) * corr;
        }

        for(size_t i=0; i<nex; i++)
        for(size_t j=0; j<nex; j++)
        {
            const Tc phase = ((i ^ j) & 1) ? Tc(-1.0) : Tc(1.0);

            const size_t ri_fixed = rows(i);
            const size_t cj_fixed = cols(j);
            const size_t mj = bit(bits, j + 2);
            
            minor_adjt(work.det_mix, work.l, i, j, work.det_mix2, work.adjt_det2,
            [&](const arma::Mat<Tc> &det_minor,
                const arma::Mat<Tc> &cof,
                const Tc det_minor_val)
            {
                const size_t lm1 = nex - 1;

                for(size_t k2=0; k2<lm1; k2++)
                {
                    const size_t k_full = minor_to_full(k2, j);
                    const size_t mk = bit(bits, k_full + 2);

                    const same_ii_slot<Tc> ii_slot =
                        resolve_same_ii_slot(II, nact, m0, m1, mk, mj);

                    const Tc corr = column_replacement_correction(
                        det_minor, cof, lm1, k2,
                        [&](const size_t r) {
                            return same_ii_replacement(
                                ii_slot,
                                rows, cols,
                                i, j,
                                r, k2,
                                ri_fixed, cj_fixed);
                        });

                    contrib += Tc(0.5) * phase * corr;
                }
            });
        }

        V += contrib;
    });
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_TWO_BODY_SAME_H
