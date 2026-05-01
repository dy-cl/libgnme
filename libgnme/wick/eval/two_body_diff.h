#ifndef LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H
#define LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H

#include <armadillo>
#include <cstdint>

#include "helpers.h"
#include "prepare.h"
#include "scratch.h"
#include "prepare_scratch.h"

namespace libgnme {
namespace wick_eval {

/** \brief Read one alpha-beta two-electron replacement entry.
    \tparam Tc Matrix element type.
    \param IIab Alpha-beta two-electron intermediate field.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param ma0 Alpha first branch selector.
    \param maj Alpha second branch selector.
    \param mb0 Beta first branch selector.
    \param mbj Beta second branch selector.
    \param ar Alpha row index.
    \param ac Alpha column index.
    \param br Beta row index.
    \param bc Beta column index.
    \return Two-electron replacement entry.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc two_body_diff_iiab(
    const arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    const size_t ma0, const size_t maj,
    const size_t mb0, const size_t mbj,
    const size_t ar, const size_t ac,
    const size_t br, const size_t bc)
{
    const arma::Mat<Tc> &M = IIab(two_body_pair(ma0,maj), two_body_pair(mb0,mbj));

    const size_t row = ac + nacta * ar;
    const size_t col = nactb * br + bc;

    return M(row,col);
}

/** \brief Evaluate different-spin two-body matrix element for nz_alpha = nz_beta = 0.
    \tparam Tc Matrix element type.
    \param rowa Alpha row indices.
    \param cola Alpha column indices.
    \param rowb Beta row indices.
    \param colb Beta column indices.
    \param V Output two-body matrix element.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    diff_scratch<Tc> &work)
{
    const size_t la = rowa.n_elem;
    const size_t lb = rowb.n_elem;

    if(la == 0 && lb == 0)
    {
        V = Vab(0,0);
        return;
    }

    if(la == 1 && lb == 1)
    {
        const size_t ra = rowa(0);
        const size_t ca = cola(0);
        const size_t rb = rowb(0);
        const size_t cb = colb(0);

        const Tc deta = work.deta0(0,0);
        const Tc detb = work.detb0(0,0);

        V = Vab(0,0) * deta * detb
          - XVbXa(0,0,0)(ra,ca) * detb
          - XVaXb(0,0,0)(rb,cb) * deta
          + two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, ra, ca, rb, cb);

        return;
    }

    Tc deta;
    Tc detb;

    if(la == 0)
    {
        work.adjt_deta.set_size(0,0);
        deta = Tc(1.0);
    }
    else
    {
        deta = adjugate_transpose(work.deta0, work.adjt_deta);
    }

    if(lb == 0)
    {
        work.adjt_detb.set_size(0,0);
        detb = Tc(1.0);
    }
    else
    {
        detb = adjugate_transpose(work.detb0, work.adjt_detb);
    }

    V = Vab(0,0) * deta * detb;

    for(size_t k=0; k<la; k++)
    {
        const size_t ca = cola(k);

        const Tc corr = column_replacement_correction(
            work.deta0, work.adjt_deta, k,
            [&](const size_t r) {
                return XVbXa(0,0,0)(rowa(r),ca);
            });

        V -= (deta + corr) * detb;
    }

    for(size_t k=0; k<lb; k++)
    {
        const size_t cb = colb(k);

        const Tc corr = column_replacement_correction(
            work.detb0, work.adjt_detb, k,
            [&](const size_t r) {
                return XVaXb(0,0,0)(rowb(r),cb);
            });

        V -= (detb + corr) * deta;
    }

    for(size_t i=0; i<la; i++)
    for(size_t j=0; j<la; j++)
    {
        const Tc detDa2 = mixed_minor_det(work.deta0, work.deta0, 0, 0, i, j, work.deta_mix_minor);
        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;

        for(size_t k=0; k<lb; k++)
        {
            const Tc corr = column_replacement_correction(
                work.detb0, work.adjt_detb, k,
                [&](const size_t r) {
                    return two_body_diff_iiab(
                        IIab, nacta, nactb,
                        0, 0, 0, 0,
                        rowa(i), cola(j),
                        rowb(r), colb(k));
                });

            V += Tc(0.5 * phase) * (detb + corr) * detDa2;
        }
    }

    for(size_t i=0; i<lb; i++)
    for(size_t j=0; j<lb; j++)
    {
        const Tc detDb2 = mixed_minor_det(work.detb0, work.detb0, 0, 0, i, j, work.detb_mix_minor);
        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;

        for(size_t k=0; k<la; k++)
        {
            const Tc corr = column_replacement_correction(
                work.deta0, work.adjt_deta, k,
                [&](const size_t r) {
                    return two_body_diff_iiab(
                        IIab, nacta, nactb,
                        0, 0, 0, 0,
                        rowa(r), cola(k),
                        rowb(i), colb(j));
                });

            V += Tc(0.5 * phase) * (deta + corr) * detDb2;
        }
    }
}

/** \brief Evaluate different-spin two-body matrix element for the generic branch.
    \tparam Tc Matrix element type.
    \param rowa Alpha row indices.
    \param cola Alpha column indices.
    \param rowb Beta row indices.
    \param colb Beta column indices.
    \param V Output two-body matrix element.
    \param nza Number of alpha zero-overlap orbital pairs.
    \param nzb Number of beta zero-overlap orbital pairs.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_gen(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const size_t &nza, const size_t &nzb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    diff_scratch<Tc> &work)
{
    const size_t la = rowa.n_elem;
    const size_t lb = rowb.n_elem;

    for_each_m_combination(la+1, nza, [&](uint64_t bitsa) {
    for_each_m_combination(lb+1, nzb, [&](uint64_t bitsb) {
        const size_t ma0 = bit(bitsa, 0);
        const size_t mb0 = bit(bitsb, 0);

        if(la == 0)
            work.deta_mix.set_size(0,0);
        else
            mix_deta(bitsa, 1, work);

        if(lb == 0)
            work.detb_mix.set_size(0,0);
        else
            mix_detb(bitsb, 1, work);

        Tc detDa;
        Tc detDb;

        if(la == 0)
        {
            detDa = Tc(1.0);
            work.adjt_deta.set_size(0,0);
        }
        else
        {
            detDa = adjugate_transpose(work.deta_mix, work.adjt_deta);
        }

        if(lb == 0)
        {
            detDb = Tc(1.0);
            work.adjt_detb.set_size(0,0);
        }
        else
        {
            detDb = adjugate_transpose(work.detb_mix, work.adjt_detb);
        }

        V += Vab(ma0,mb0) * detDa * detDb;

        for(size_t i=0; i<la; i++)
        {
            const size_t mai = bit(bitsa, i+1);

            const Tc corr = column_replacement_correction(
                work.deta_mix, work.adjt_deta, i,
                [&](const size_t r) {
                    return XVbXa(ma0,mb0,mai)(rowa(r),cola(i));
                });

            V -= (detDa + corr) * detDb;
        }

        for(size_t i=0; i<lb; i++)
        {
            const size_t mbi = bit(bitsb, i+1);

            const Tc corr = column_replacement_correction(
                work.detb_mix, work.adjt_detb, i,
                [&](const size_t r) {
                    return XVaXb(mb0,ma0,mbi)(rowb(r),colb(i));
                });

            V -= (detDb + corr) * detDa;
        }

        for(size_t i=0; i<la; i++)
        for(size_t j=0; j<la; j++)
        {
            const Tc detDa2 = mixed_minor_det(
                work.deta0, work.deta1, bitsa, 2, i, j, work.deta_mix_minor);

            const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;
            const size_t maj = bit(bitsa, 1);

            for(size_t k=0; k<lb; k++)
            {
                const size_t mbk = bit(bitsb, k+1);

                const Tc corr = column_replacement_correction(
                    work.detb_mix, work.adjt_detb, k,
                    [&](const size_t r) {
                        return two_body_diff_iiab(
                            IIab, nacta, nactb,
                            ma0, maj,
                            mb0, mbk,
                            rowa(i), cola(j),
                            rowb(r), colb(k));
                    });

                V += Tc(0.5 * phase) * (detDb + corr) * detDa2;
            }
        }

        for(size_t i=0; i<lb; i++)
        for(size_t j=0; j<lb; j++)
        {
            const Tc detDb2 = mixed_minor_det(
                work.detb0, work.detb1, bitsb, 2, i, j, work.detb_mix_minor);

            const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;
            const size_t mbj = bit(bitsb, 1);

            for(size_t k=0; k<la; k++)
            {
                const size_t mak = bit(bitsa, k+1);

                const Tc corr = column_replacement_correction(
                    work.deta_mix, work.adjt_deta, k,
                    [&](const size_t r) {
                        return two_body_diff_iiab(
                            IIab, nacta, nactb,
                            ma0, mak,
                            mb0, mbj,
                            rowa(r), cola(k),
                            rowb(i), colb(j));
                    });

                V += Tc(0.5 * phase) * (detDa + corr) * detDb2;
            }
        }
    });
    });
}

/** \brief Evaluate different-spin two-body contribution using Wick contractions.
    \tparam Tc Matrix element type.
    \param xahp Alpha particle-hole indices for bra state.
    \param xbhp Beta particle-hole indices for bra state.
    \param wahp Alpha particle-hole indices for ket state.
    \param wbhp Beta particle-hole indices for ket state.
    \param V Output two-body matrix element.
    \param nza Number of alpha zero-overlap orbital pairs.
    \param nzb Number of beta zero-overlap orbital pairs.
    \param Xa Alpha lower-triangular contractions.
    \param Ya Alpha upper-triangular contractions.
    \param Xb Beta lower-triangular contractions.
    \param Yb Beta upper-triangular contractions.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nactxa Alpha bra active dimension.
    \param nactwa Alpha ket active dimension.
    \param nactxb Beta bra active dimension.
    \param nactwb Beta ket active dimension.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff(
    arma::umat xahp, arma::umat xbhp,
    arma::umat wahp, arma::umat wbhp,
    Tc &V,
    const size_t &nza, const size_t &nzb,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Ya,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::field<arma::Mat<Tc> > &Yb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nactxa,
    const size_t nactwa,
    const size_t nactxb,
    const size_t nactwb)
{
    V = Tc(0.0);

    const size_t nxa = xahp.n_rows;
    const size_t nwa = wahp.n_rows;
    const size_t nxb = xbhp.n_rows;
    const size_t nwb = wbhp.n_rows;

    const size_t la = nxa + nwa;
    const size_t lb = nxb + nwb;

    if(nza > la + 1 || nzb > lb + 1) return;

    scratch<Tc> &store = local_scratch<Tc>();
    diff_scratch<Tc> &work = store.diff;

    arma::uvec rowa, cola;
    arma::uvec rowb, colb;

    prepare_diff(
        xahp, wahp, xbhp, wbhp,
        nactxa, nactxb,
        nza, nzb,
        Xa, Ya, Xb, Yb,
        rowa, cola, rowb, colb,
        work);

    const size_t nacta = nactxa + nactwa;
    const size_t nactb = nactxb + nactwb;

    if(nza == 0 && nzb == 0)
    {
        two_body_diff_m0(
            rowa, cola, rowb, colb, V,
            Vab, XVaXb, XVbXa, IIab,
            nacta, nactb, work);
        return;
    }

    two_body_diff_gen(
        rowa, cola, rowb, colb, V,
        nza, nzb,
        Vab, XVaXb, XVbXa, IIab,
        nacta, nactb, work);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H
