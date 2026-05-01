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

/** \brief Evaluate different-spin two-body contribution using prepared Wick contractions.
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param nza Number of alpha zero-overlap orbital pairs.
    \param nzb Number of beta zero-overlap orbital pairs.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param a Prepared alpha same-spin scratch storage.
    \param b Prepared beta same-spin scratch storage.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff(
    Tc &V,
    const size_t &nza,
    const size_t &nzb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    same_scratch<Tc> &a,
    same_scratch<Tc> &b,
    diff_scratch<Tc> &work)
{
    V = Tc(0.0);

    const size_t la = a.rows.n_elem;
    const size_t lb = b.rows.n_elem;

    if(nza > la + 1 || nzb > lb + 1) return;

    work.ensure(la, lb);

    if(nza == 0 && nzb == 0)
    {
        two_body_diff_m0(V, Vab, XVaXb, XVbXa, IIab, nacta, nactb, a, b, work);
        return;
    }

    two_body_diff_gen(
        a.rows, a.cols, b.rows, b.cols, V,
        nza, nzb,
        Vab, XVaXb, XVbXa, IIab,
        nacta, nactb,
        a, b, work);
}

/** \brief Evaluate different-spin two-body matrix element for nz_alpha = nz_beta = 0.
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param a Prepared alpha same-spin scratch storage.
    \param b Prepared beta same-spin scratch storage.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0(
    Tc &V,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    same_scratch<Tc> &a,
    same_scratch<Tc> &b,
    diff_scratch<Tc> &work)
{
    const arma::uvec &rowa = a.rows;
    const arma::uvec &cola = a.cols;
    const arma::uvec &rowb = b.rows;
    const arma::uvec &colb = b.cols;

    const size_t la = rowa.n_elem;
    const size_t lb = rowb.n_elem;

    if(la == 0 && lb == 0)
    {
        V = Vab(0,0);
        return;
    }

    if(la == 1 && lb == 1)
    {
        two_body_diff_m0_11(
            V, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb, a, b);
        return;
    }

    if(la == 1 && lb == 3)
    {
        two_body_diff_m0_13(
            V, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb, a, b, work);
        return;
    }

    if(la == 3 && lb == 1)
    {
        two_body_diff_m0_31(
            V, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb, a, b, work);
        return;
    }

    if(la == 2 && lb == 2)
    {
        two_body_diff_m0_22(
            V, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb, a, b);
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
        deta = adjugate_transpose(a.det0, work.adjt_deta);
    }

    if(lb == 0)
    {
        work.adjt_detb.set_size(0,0);
        detb = Tc(1.0);
    }
    else
    {
        detb = adjugate_transpose(b.det0, work.adjt_detb);
    }

    V = Vab(0,0) * deta * detb;

    const arma::Mat<Tc> &XVa = XVbXa(0,0,0);
    const arma::Mat<Tc> &XVb = XVaXb(0,0,0);

    for(size_t c=0; c<la; c++)
    for(size_t r=0; r<la; r++)
    {
        V -= work.adjt_deta(r,c) * XVa(rowa(r), cola(c)) * detb;
    }

    for(size_t c=0; c<lb; c++)
    for(size_t r=0; r<lb; r++)
    {
        V -= work.adjt_detb(r,c) * XVb(rowb(r), colb(c)) * deta;
    }

    for(size_t cb=0; cb<lb; cb++)
    for(size_t rb=0; rb<lb; rb++)
    for(size_t ca=0; ca<la; ca++)
    for(size_t ra=0; ra<la; ra++)
    {
        V += work.adjt_deta(ra,ca)
           * work.adjt_detb(rb,cb)
           * two_body_diff_iiab(
                 IIab, nacta, nactb,
                 0, 0, 0, 0,
                 rowa(ra), cola(ca),
                 rowb(rb), colb(cb));
    }
}

/** \brief Evaluate different-spin two-body matrix element for nz = 0 and ranks (1,1).
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param a Prepared alpha same-spin scratch storage.
    \param b Prepared beta same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_11(
    Tc &V,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    same_scratch<Tc> &a,
    same_scratch<Tc> &b)
{
    const size_t ra = a.rows(0);
    const size_t ca = a.cols(0);
    const size_t rb = b.rows(0);
    const size_t cb = b.cols(0);

    const Tc deta = a.det0(0,0);
    const Tc detb = b.det0(0,0);

    V = Vab(0,0) * deta * detb
      - XVbXa(0,0,0)(ra,ca) * detb
      - XVaXb(0,0,0)(rb,cb) * deta
      + two_body_diff_iiab(
            IIab, nacta, nactb,
            0, 0, 0, 0,
            ra, ca, rb, cb);
}

/** \brief Evaluate different-spin two-body matrix element for nz = 0 and ranks (2,2).
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param a Prepared alpha same-spin scratch storage.
    \param b Prepared beta same-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_22(
    Tc &V,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    same_scratch<Tc> &a,
    same_scratch<Tc> &b)
{
    const Tc a00 = a.det0(0,0);
    const Tc a01 = a.det0(0,1);
    const Tc a10 = a.det0(1,0);
    const Tc a11 = a.det0(1,1);
    const Tc deta = a00 * a11 - a01 * a10;

    const Tc b00 = b.det0(0,0);
    const Tc b01 = b.det0(0,1);
    const Tc b10 = b.det0(1,0);
    const Tc b11 = b.det0(1,1);
    const Tc detb = b00 * b11 - b01 * b10;

    const size_t r0a = a.rows(0);
    const size_t r1a = a.rows(1);
    const size_t c0a = a.cols(0);
    const size_t c1a = a.cols(1);

    const size_t r0b = b.rows(0);
    const size_t r1b = b.rows(1);
    const size_t c0b = b.cols(0);
    const size_t c1b = b.cols(1);

    const arma::Mat<Tc> &A = XVbXa(0,0,0);
    const arma::Mat<Tc> &B = XVaXb(0,0,0);

    const Tc au0 = A(r0a,c0a);
    const Tc au1 = A(r1a,c0a);
    const Tc av0 = A(r0a,c1a);
    const Tc av1 = A(r1a,c1a);
    const Tc deta_c0 = au0 * a11 - a01 * au1;
    const Tc deta_c1 = a00 * av1 - av0 * a10;

    const Tc bu0 = B(r0b,c0b);
    const Tc bu1 = B(r1b,c0b);
    const Tc bv0 = B(r0b,c1b);
    const Tc bv1 = B(r1b,c1b);
    const Tc detb_c0 = bu0 * b11 - b01 * bu1;
    const Tc detb_c1 = b00 * bv1 - bv0 * b10;

    V = Vab(0,0) * deta * detb
      - (deta_c0 + deta_c1) * detb
      - (detb_c0 + detb_c1) * deta;

    const Tc cofa00 = a11;
    const Tc cofa10 = -a10;
    const Tc cofa01 = -a01;
    const Tc cofa11 = a00;

    const Tc cofb00 = b11;
    const Tc cofb10 = -b10;
    const Tc cofb01 = -b01;
    const Tc cofb11 = b00;

    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<2; j++)
    {
        const size_t ra = a.rows(i);
        const size_t ca = a.cols(j);

        const Tc cof =
            (i == 0 && j == 0) ? cofa00 :
            (i == 1 && j == 0) ? cofa10 :
            (i == 0 && j == 1) ? cofa01 : cofa11;

        const Tc x00 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, ra, ca, r0b, c0b);
        const Tc x10 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, ra, ca, r1b, c0b);
        const Tc x01 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, ra, ca, r0b, c1b);
        const Tc x11 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, ra, ca, r1b, c1b);

        V += Tc(0.5) * cof * (x00 * b11 - b01 * x10);
        V += Tc(0.5) * cof * (b00 * x11 - x01 * b10);
    }

    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<2; j++)
    {
        const size_t rb = b.rows(i);
        const size_t cb = b.cols(j);

        const Tc cof =
            (i == 0 && j == 0) ? cofb00 :
            (i == 1 && j == 0) ? cofb10 :
            (i == 0 && j == 1) ? cofb01 : cofb11;

        const Tc x00 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, r0a, c0a, rb, cb);
        const Tc x10 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, r1a, c0a, rb, cb);
        const Tc x01 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, r0a, c1a, rb, cb);
        const Tc x11 = two_body_diff_iiab(IIab, nacta, nactb, 0, 0, 0, 0, r1a, c1a, rb, cb);

        V += Tc(0.5) * cof * (x00 * a11 - a01 * x10);
        V += Tc(0.5) * cof * (a00 * x11 - x01 * a10);
    }
}

/** \brief Evaluate different-spin two-body matrix element for nz = 0 and ranks (1,3).
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param a Prepared alpha same-spin scratch storage.
    \param b Prepared beta same-spin scratch storage.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_13(
    Tc &V,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    same_scratch<Tc> &a,
    same_scratch<Tc> &b,
    diff_scratch<Tc> &work)
{
    const size_t ra = a.rows(0);
    const size_t ca = a.cols(0);
    const Tc deta = a.det0(0,0);

    const Tc detb = adjugate_transpose(b.det0, work.adjt_detb);
    const arma::Mat<Tc> &cofb = work.adjt_detb;

    const arma::Mat<Tc> &A = XVbXa(0,0,0);
    const arma::Mat<Tc> &B = XVaXb(0,0,0);

    Tc beta_term = Tc(0.0);
    for(size_t c=0; c<3; c++)
    for(size_t r=0; r<3; r++)
        beta_term += cofb(r,c) * B(b.rows(r), b.cols(c));

    Tc ii_term = Tc(0.0);
    for(size_t c=0; c<3; c++)
    for(size_t r=0; r<3; r++)
        ii_term += cofb(r,c) * two_body_diff_iiab(
            IIab, nacta, nactb, 0, 0, 0, 0,
            ra, ca, b.rows(r), b.cols(c));

    V = Vab(0,0) * deta * detb
      - A(ra,ca) * detb
      - deta * beta_term
      + ii_term;
}

/** \brief Evaluate different-spin two-body matrix element for nz = 0 and ranks (3,1).
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param Vab Zeroth-order alpha-beta contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \param a Prepared alpha same-spin scratch storage.
    \param b Prepared beta same-spin scratch storage.
    \param work Different-spin scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_31(
    Tc &V,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    same_scratch<Tc> &a,
    same_scratch<Tc> &b,
    diff_scratch<Tc> &work)
{
    const size_t rb = b.rows(0);
    const size_t cb = b.cols(0);
    const Tc detb = b.det0(0,0);

    const Tc deta = adjugate_transpose(a.det0, work.adjt_deta);
    const arma::Mat<Tc> &cofa = work.adjt_deta;

    const arma::Mat<Tc> &A = XVbXa(0,0,0);
    const arma::Mat<Tc> &B = XVaXb(0,0,0);

    Tc alpha_term = Tc(0.0);
    for(size_t c=0; c<3; c++)
    for(size_t r=0; r<3; r++)
        alpha_term += cofa(r,c) * A(a.rows(r), a.cols(c));

    Tc ii_term = Tc(0.0);
    for(size_t c=0; c<3; c++)
    for(size_t r=0; r<3; r++)
        ii_term += cofa(r,c) * two_body_diff_iiab(
            IIab, nacta, nactb, 0, 0, 0, 0,
            a.rows(r), a.cols(c), rb, cb);

    V = Vab(0,0) * deta * detb
      - detb * alpha_term
      - B(rb,cb) * deta
      + ii_term;
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
    const size_t &nza,
    const size_t &nzb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb,
    same_scratch<Tc> &a,
    same_scratch<Tc> &b,
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
            mix_deta(bitsa, 1, a, work);

        if(lb == 0)
            work.detb_mix.set_size(0,0);
        else
            mix_detb(bitsb, 1, b, work);

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
                a.det0, a.det1, bitsa, 2, i, j, work.deta_mix_minor);

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
                b.det0, b.det1, bitsb, 2, i, j, work.detb_mix_minor);

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

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H
