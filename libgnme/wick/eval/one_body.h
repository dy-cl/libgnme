#ifndef LIBGNME_WICK_EVAL_ONE_BODY_H
#define LIBGNME_WICK_EVAL_ONE_BODY_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "prepare.h"
#include "scratch.h"
#include "prepare_scratch.h"

namespace libgnme {
namespace wick_eval {

/** \brief Evaluate one-body matrix element for nz = 0 and zero excitations.
    \tparam Tc Matrix element type.
    \param F Output one-body matrix element.
    \param F0 Zeroth-order one-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_l0(Tc &F, const arma::Col<Tc> &F0)
{
    F = F0(0);
}

/** \brief Evaluate one-body matrix element for nz = 0 and one excitation.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_l1(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX,
    same_scratch<Tc> &work)
{
    const size_t r0 = rows(0);
    const size_t c0 = cols(0);

    F = work.det0(0,0) * F0(0) - XFX(0,0)(r0,c0);
}

/** \brief Evaluate one-body matrix element for nz = 0 and two excitations.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_l2(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX,
    same_scratch<Tc> &work)
{
    const Tc a00 = work.det0(0,0);
    const Tc a01 = work.det0(0,1);
    const Tc a10 = work.det0(1,0);
    const Tc a11 = work.det0(1,1);

    const Tc detD = det2_scalar(a00, a01, a10, a11);

    const arma::Mat<Tc> &Fmat = XFX(0,0);

    const size_t r0 = rows(0);
    const size_t r1 = rows(1);
    const size_t c0 = cols(0);
    const size_t c1 = cols(1);

    const Tc u0 = Fmat(r0,c0);
    const Tc u1 = Fmat(r1,c0);
    const Tc v0 = Fmat(r0,c1);
    const Tc v1 = Fmat(r1,c1);

    const Tc det_c0 = u0 * a11 - a01 * u1;
    const Tc det_c1 = a00 * v1 - v0 * a10;

    F = detD * F0(0) - det_c0 - det_c1;
}

/** \brief Evaluate one-body matrix element for nz = 0 and arbitrary excitation rank.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX,
    same_scratch<Tc> &work)
{
    const size_t nex = rows.n_elem;

    const Tc detD = adjugate_transpose(work.det0, work.adjt_det);

    F = F0(0) * detD;

    const arma::Mat<Tc> &Fmat = XFX(0,0);

    for(size_t k=0; k<nex; k++)
    {
        const size_t ck = cols(k);

        const Tc corr = column_replacement_correction(
            work.det0, work.adjt_det, k,
            [&](const size_t r) {
                return Fmat(rows(r), ck);
            });

        F -= (detD + corr);
    }
}

/** \brief Dispatch one-body matrix element for nz = 0.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_m0(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX,
    same_scratch<Tc> &work)
{
    const size_t nex = rows.n_elem;

    if(nex == 0)
    {
        one_body_m0_l0(F, F0);
        return;
    }

    if(nex == 1)
    {
        one_body_m0_l1(rows, cols, F, F0, XFX, work);
        return;
    }

    if(nex == 2)
    {
        one_body_m0_l2(rows, cols, F, F0, XFX, work);
        return;
    }

    one_body_m0_gen(rows, cols, F, F0, XFX, work);
}

/** \brief Evaluate one-body matrix element for the generic nz > 0 case.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param F Output one-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \param work Same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void one_body_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &F,
    const size_t &nz,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX,
    same_scratch<Tc> &work)
{
    const size_t nex = rows.n_elem;

    if(nex == 0)
    {
        for_each_m_combination(1, nz, [&](uint64_t bits) {
            const size_t m0 = bit(bits, 0);
            F += F0(m0);
        });

        return;
    }

    if(nex == 1)
    {
        const size_t r0 = rows(0);
        const size_t c0 = cols(0);

        for_each_m_combination(2, nz, [&](uint64_t bits) {
            const size_t m0 = bit(bits, 0);
            const size_t m1 = bit(bits, 1);

            F += (m1 ? work.det1(0,0) : work.det0(0,0)) * F0(m0)
               - XFX(m0,m1)(r0,c0);
        });

        return;
    }

    mix_dets_same(nex, nz, 1, work, [&](const uint64_t bits) {
        const size_t m0 = bit(bits, 0);

        const Tc detD = adjugate_transpose(work.det_mix, work.adjt_det);

        F += F0(m0) * detD;

        for(size_t k=0; k<nex; k++)
        {
            const size_t mk = bit(bits, k+1);
            const size_t ck = cols(k);

            const Tc corr = column_replacement_correction(
                work.det_mix, work.adjt_det, k,
                [&](const size_t r) {
                    return XFX(m0,mk)(rows(r), ck);
                });

            F -= (detD + corr);
        }
    });
}

/** \brief Evaluate one-body same-spin contribution using prepared Wick contractions.
    \tparam Tc Matrix element type.
    \param F Output one-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param F0 Zeroth-order one-body contractions.
    \param XFX First-order one-body contractions.
    \param work Prepared same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void spin_one_body(
    Tc &F,
    const size_t &nz,
    const arma::Col<Tc> &F0,
    const arma::field<arma::Mat<Tc> > &XFX,
    same_scratch<Tc> &work)
{
    F = Tc(0.0);

    const size_t nex = work.rows.n_elem;
    if(nz > nex + 1) return;

    const arma::uvec &rows = work.rows;
    const arma::uvec &cols = work.cols;

    if(nz == 0)
    {
        one_body_m0(rows, cols, F, F0, XFX, work);
        return;
    }

    one_body_gen(rows, cols, F, nz, F0, XFX, work);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_ONE_BODY_H
