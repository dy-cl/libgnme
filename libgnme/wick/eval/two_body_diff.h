#ifndef LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H
#define LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H

#include <armadillo>
#include <cstdint>
#include <libgnme/utils/linalg.h>

#include "helpers.h"
#include "prepare.h"

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

/** \brief Evaluate different-spin two-body matrix element for nza = nzb = 0 and no excitations.
    \tparam Tc Matrix element type.
    \param V Output two-body matrix element.
    \param Vab Zeroth-order alpha-beta contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_00(Tc &V, const arma::Mat<Tc> &Vab)
{
    V = Vab(0,0);
}

/** \brief Evaluate different-spin two-body matrix element for nza = nzb = 0 and la = lb = 1.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_11(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb)
{
    const size_t ra = rowa(0);
    const size_t ca = cola(0);
    const size_t rb = rowb(0);
    const size_t cb = colb(0);

    const Tc deta = Xa(0)(ra,ca);
    const Tc detb = Xb(0)(rb,cb);

    V = Vab(0,0) * deta * detb
      - XVbXa(0,0,0)(ra,ca) * detb
      - XVaXb(0,0,0)(rb,cb) * deta
      + two_body_diff_iiab(
            IIab, nacta, nactb,
            0, 0, 0, 0,
            ra, ca, rb, cb);
}

/** \brief Evaluate different-spin two-body matrix element for nza = nzb = 0 and la = 1, lb = 3.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_13(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::field<arma::Mat<Tc> > &Yb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb)
{
    const size_t ra = rowa(0);
    const size_t ca = cola(0);

    const Tc deta = Xa(0)(ra,ca);

    arma::Mat<Tc> Db;
    build_det(Xb(0), Yb(0), rowb, colb, Db);

    arma::Mat<Tc> cofB;
    const Tc detb = adjugate_transpose(Db, cofB);

    const Tc alpha_repl = XVbXa(0,0,0)(ra,ca);

    Tc beta_repl = Tc(0.0);
    Tc ii_repl = Tc(0.0);

    for(size_t k=0; k<3; k++)
    {
        const size_t cb = colb(k);

        for(size_t r=0; r<3; r++)
        {
            const size_t rb = rowb(r);

            beta_repl += cofB(r,k) * XVaXb(0,0,0)(rb,cb);

            ii_repl += cofB(r,k) * two_body_diff_iiab(
                IIab, nacta, nactb,
                0, 0, 0, 0,
                ra, ca, rb, cb);
        }
    }

    V = Vab(0,0) * deta * detb
      - alpha_repl * detb
      - deta * beta_repl
      + ii_repl;
}

/** \brief Evaluate different-spin two-body matrix element for nza = nzb = 0 and la = 3, lb = 1.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_31(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Ya,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb)
{
    const size_t rb = rowb(0);
    const size_t cb = colb(0);

    const Tc detb = Xb(0)(rb,cb);

    arma::Mat<Tc> Da;
    build_det(Xa(0), Ya(0), rowa, cola, Da);

    arma::Mat<Tc> cofA;
    const Tc deta = adjugate_transpose(Da, cofA);

    const Tc beta_repl = XVaXb(0,0,0)(rb,cb);

    Tc alpha_repl = Tc(0.0);
    Tc ii_repl = Tc(0.0);

    for(size_t k=0; k<3; k++)
    {
        const size_t ca = cola(k);

        for(size_t r=0; r<3; r++)
        {
            const size_t ra = rowa(r);

            alpha_repl += cofA(r,k) * XVbXa(0,0,0)(ra,ca);

            ii_repl += cofA(r,k) * two_body_diff_iiab(
                IIab, nacta, nactb,
                0, 0, 0, 0,
                ra, ca, rb, cb);
        }
    }

    V = Vab(0,0) * deta * detb
      - alpha_repl * detb
      - beta_repl * deta
      + ii_repl;
}

/** \brief Evaluate different-spin two-body matrix element for nza = nzb = 0 and la = lb = 2.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_22(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Ya,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::field<arma::Mat<Tc> > &Yb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb)
{
    arma::Mat<Tc> Da;
    arma::Mat<Tc> Db;

    build_det(Xa(0), Ya(0), rowa, cola, Da);
    build_det(Xb(0), Yb(0), rowb, colb, Db);

    arma::Mat<Tc> cofA;
    arma::Mat<Tc> cofB;

    const Tc deta = adjugate_transpose(Da, cofA);
    const Tc detb = adjugate_transpose(Db, cofB);

    Tc alpha_repl = Tc(0.0);
    Tc beta_repl = Tc(0.0);

    for(size_t k=0; k<2; k++)
    {
        const size_t ca = cola(k);

        for(size_t r=0; r<2; r++)
            alpha_repl += cofA(r,k) * XVbXa(0,0,0)(rowa(r),ca);
    }

    for(size_t k=0; k<2; k++)
    {
        const size_t cb = colb(k);

        for(size_t r=0; r<2; r++)
            beta_repl += cofB(r,k) * XVaXb(0,0,0)(rowb(r),cb);
    }

    Tc ii_repl = Tc(0.0);

    for(size_t ia=0; ia<2; ia++)
    for(size_t ja=0; ja<2; ja++)
    {
        const size_t ra = rowa(ia);
        const size_t ca = cola(ja);
        const Tc cofa = cofA(ia,ja);

        for(size_t kb=0; kb<2; kb++)
        {
            const size_t cb = colb(kb);

            for(size_t rb_i=0; rb_i<2; rb_i++)
            {
                const size_t rb = rowb(rb_i);

                ii_repl += Tc(0.5) * cofa * cofB(rb_i,kb) *
                    two_body_diff_iiab(
                        IIab, nacta, nactb,
                        0, 0, 0, 0,
                        ra, ca, rb, cb);
            }
        }
    }

    for(size_t ib=0; ib<2; ib++)
    for(size_t jb=0; jb<2; jb++)
    {
        const size_t rb = rowb(ib);
        const size_t cb = colb(jb);
        const Tc cofb = cofB(ib,jb);

        for(size_t ka=0; ka<2; ka++)
        {
            const size_t ca = cola(ka);

            for(size_t ra_i=0; ra_i<2; ra_i++)
            {
                const size_t ra = rowa(ra_i);

                ii_repl += Tc(0.5) * cofb * cofA(ra_i,ka) *
                    two_body_diff_iiab(
                        IIab, nacta, nactb,
                        0, 0, 0, 0,
                        ra, ca, rb, cb);
            }
        }
    }

    V = Vab(0,0) * deta * detb
      - alpha_repl * detb
      - beta_repl * deta
      + ii_repl;
}

/** \brief Evaluate different-spin two-body matrix element for nza = nzb = 0 and arbitrary ranks.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0_gen(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Ya,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::field<arma::Mat<Tc> > &Yb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb)
{
    const size_t la = rowa.n_elem;
    const size_t lb = rowb.n_elem;

    arma::Mat<Tc> Da, Db;
    arma::Mat<Tc> cofA, cofB;

    Tc deta;
    Tc detb;

    if(la == 0)
    {
        Da.set_size(0,0);
        cofA.set_size(0,0);
        deta = Tc(1.0);
    }
    else
    {
        build_det(Xa(0), Ya(0), rowa, cola, Da);
        deta = adjugate_transpose(Da, cofA);
    }

    if(lb == 0)
    {
        Db.set_size(0,0);
        cofB.set_size(0,0);
        detb = Tc(1.0);
    }
    else
    {
        build_det(Xb(0), Yb(0), rowb, colb, Db);
        detb = adjugate_transpose(Db, cofB);
    }

    V = Vab(0,0) * deta * detb;

    for(size_t k=0; k<la; k++)
    {
        const size_t ca = cola(k);

        const Tc corr = column_replacement_correction(
            Da, cofA, k,
            [&](const size_t r) {
                return XVbXa(0,0,0)(rowa(r),ca);
            });

        V -= (deta + corr) * detb;
    }

    for(size_t k=0; k<lb; k++)
    {
        const size_t cb = colb(k);

        const Tc corr = column_replacement_correction(
            Db, cofB, k,
            [&](const size_t r) {
                return XVaXb(0,0,0)(rowb(r),cb);
            });

        V -= (detb + corr) * deta;
    }

    arma::Mat<Tc> minorD;

    for(size_t i=0; i<la; i++)
    for(size_t j=0; j<la; j++)
    {
        const Tc detDa2 = mixed_minor_det(Da, Da, 0, 0, i, j, minorD);
        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;

        for(size_t k=0; k<lb; k++)
        {
            const size_t cb = colb(k);

            const Tc corr = column_replacement_correction(
                Db, cofB, k,
                [&](const size_t r) {
                    return two_body_diff_iiab(
                        IIab, nacta, nactb,
                        0, 0, 0, 0,
                        rowa(i), cola(j),
                        rowb(r), cb);
                });

            V += Tc(0.5 * phase) * (detb + corr) * detDa2;
        }
    }

    for(size_t i=0; i<lb; i++)
    for(size_t j=0; j<lb; j++)
    {
        const Tc detDb2 = mixed_minor_det(Db, Db, 0, 0, i, j, minorD);
        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;

        for(size_t k=0; k<la; k++)
        {
            const size_t ca = cola(k);

            const Tc corr = column_replacement_correction(
                Da, cofA, k,
                [&](const size_t r) {
                    return two_body_diff_iiab(
                        IIab, nacta, nactb,
                        0, 0, 0, 0,
                        rowa(r), ca,
                        rowb(i), colb(j));
                });

            V += Tc(0.5 * phase) * (deta + corr) * detDb2;
        }
    }
}

/** \brief Dispatch different-spin two-body matrix element for nza = nzb = 0.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_m0(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &Xa,
    const arma::field<arma::Mat<Tc> > &Ya,
    const arma::field<arma::Mat<Tc> > &Xb,
    const arma::field<arma::Mat<Tc> > &Yb,
    const arma::Mat<Tc> &Vab,
    const arma::field<arma::Mat<Tc> > &XVaXb,
    const arma::field<arma::Mat<Tc> > &XVbXa,
    arma::field<arma::Mat<Tc> > &IIab,
    const size_t nacta,
    const size_t nactb)
{
    const size_t la = rowa.n_elem;
    const size_t lb = rowb.n_elem;

    if(la == 0 && lb == 0)
    {
        two_body_diff_m0_00(V, Vab);
        return;
    }

    if(la == 1 && lb == 1)
    {
        two_body_diff_m0_11(
            rowa, cola, rowb, colb, V,
            Xa, Xb, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb);
        return;
    }

    if(la == 1 && lb == 3)
    {
        two_body_diff_m0_13(
            rowa, cola, rowb, colb, V,
            Xa, Xb, Yb, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb);
        return;
    }

    if(la == 2 && lb == 2)
    {
        two_body_diff_m0_22(
            rowa, cola, rowb, colb, V,
            Xa, Ya, Xb, Yb, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb);
        return;
    }

    if(la == 3 && lb == 1)
    {
        two_body_diff_m0_31(
            rowa, cola, rowb, colb, V,
            Xa, Ya, Xb, Vab, XVaXb, XVbXa, IIab,
            nacta, nactb);
        return;
    }

    two_body_diff_m0_gen(
        rowa, cola, rowb, colb, V,
        Xa, Ya, Xb, Yb,
        Vab, XVaXb, XVbXa, IIab,
        nacta, nactb);
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
    \param Xa Alpha lower-triangular contractions.
    \param Ya Alpha upper-triangular contractions.
    \param Xb Beta lower-triangular contractions.
    \param Yb Beta upper-triangular contractions.
    \param Vab Zeroth-order alpha-beta two-body contractions.
    \param XVaXb First-order alpha-beta contractions.
    \param XVbXa First-order beta-alpha contractions.
    \param IIab Alpha-beta two-electron intermediates.
    \param nacta Total alpha active dimension.
    \param nactb Total beta active dimension.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_diff_gen(
    const arma::uvec &rowa, const arma::uvec &cola,
    const arma::uvec &rowb, const arma::uvec &colb,
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
    const size_t nacta,
    const size_t nactb)
{
    const size_t la = rowa.n_elem;
    const size_t lb = rowb.n_elem;

    const size_t da = (nza > 0) ? 2 : 1;
    const size_t db = (nzb > 0) ? 2 : 1;

    arma::Mat<Tc> Da, DaB;
    if(la == 0)
    {
        Da.set_size(0,0);
        DaB.set_size(0,0);
    }
    else
    {
        build_det(Xa(0), Ya(0), rowa, cola, Da);
        build_det(Xa(1), Ya(1), rowa, cola, DaB);
    }

    arma::Mat<Tc> Db, DbB;
    if(lb == 0)
    {
        Db.set_size(0,0);
        DbB.set_size(0,0);
    }
    else
    {
        build_det(Xb(0), Yb(0), rowb, colb, Db);
        build_det(Xb(1), Yb(1), rowb, colb, DbB);
    }

    arma::field<arma::Mat<Tc> > Jab(db,da,db), Jba(da,db,da);

    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
    for(size_t k=0; k<db; k++)
        build_mat(XVbXa(i,k,j), rowa, cola, Jba(i,k,j));

    for(size_t i=0; i<db; i++)
    for(size_t j=0; j<db; j++)
    for(size_t k=0; k<da; k++)
        build_mat(XVaXb(i,k,j), rowb, colb, Jab(i,k,j));

    arma::Mat<Tc> tmpDa, tmpDb;
    arma::Mat<Tc> adjDa, adjDb;
    arma::Mat<Tc> minorD;

    for_each_m_combination(la+1, nza, [&](uint64_t bitsa) {
    for_each_m_combination(lb+1, nzb, [&](uint64_t bitsb) {
        const size_t ma0 = bit(bitsa, 0);
        const size_t mb0 = bit(bitsb, 0);

        if(la == 0)
        {
            tmpDa.set_size(0,0);
        }
        else
        {
            mix_det(Da, DaB, bitsa, 1, tmpDa);
        }

        if(lb == 0)
        {
            tmpDb.set_size(0,0);
        }
        else
        {
            mix_det(Db, DbB, bitsb, 1, tmpDb);
        }

        Tc detDa;
        Tc detDb;

        if(la == 0)
        {
            detDa = Tc(1.0);
            adjDa.set_size(0,0);
        }
        else
        {
            detDa = adjugate_transpose(tmpDa, adjDa);
        }

        if(lb == 0)
        {
            detDb = Tc(1.0);
            adjDb.set_size(0,0);
        }
        else
        {
            detDb = adjugate_transpose(tmpDb, adjDb);
        }

        V += Vab(ma0,mb0) * detDa * detDb;

        for(size_t i=0; i<la; i++)
        {
            const size_t mai = bit(bitsa, i+1);

            arma::Col<Tc> v1(Jba(ma0,mb0,mai).colptr(i), la, false, true);
            arma::Col<Tc> v2(tmpDa.colptr(i), la, false, true);
            arma::Col<Tc> a(adjDa.colptr(i), la, false, true);

            V -= (detDa + arma::dot(v1-v2,a)) * detDb;
        }

        for(size_t i=0; i<lb; i++)
        {
            const size_t mbi = bit(bitsb, i+1);

            arma::Col<Tc> v1(Jab(mb0,ma0,mbi).colptr(i), lb, false, true);
            arma::Col<Tc> v2(tmpDb.colptr(i), lb, false, true);
            arma::Col<Tc> a(adjDb.colptr(i), lb, false, true);

            V -= (detDb + arma::dot(v1-v2,a)) * detDa;
        }

        for(size_t i=0; i<la; i++)
        for(size_t j=0; j<la; j++)
        {
            const Tc detDa2 = mixed_minor_det(Da, DaB, bitsa, 2, i, j, minorD);
            const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;
            const size_t maj = bit(bitsa, 1);

            for(size_t k=0; k<lb; k++)
            {
                const size_t mbk = bit(bitsb, k+1);

                const Tc corr = column_replacement_correction(
                    tmpDb, adjDb, k,
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
            const Tc detDb2 = mixed_minor_det(Db, DbB, bitsb, 2, i, j, minorD);
            const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;
            const size_t mbj = bit(bitsb, 1);

            for(size_t k=0; k<la; k++)
            {
                const size_t mak = bit(bitsa, k+1);

                const Tc corr = column_replacement_correction(
                    tmpDa, adjDa, k,
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

    wahp += nactxa;
    wbhp += nactxb;

    arma::uvec rowa, cola;
    arma::uvec rowb, colb;

    indices(xahp, wahp, rowa, cola);
    indices(xbhp, wbhp, rowb, colb);

    const size_t nacta = nactxa + nactwa;
    const size_t nactb = nactxb + nactwb;

    if(nza == 0 && nzb == 0)
    {
        two_body_diff_m0(
            rowa, cola, rowb, colb, V,
            Xa, Ya, Xb, Yb,
            Vab, XVaXb, XVbXa, IIab,
            nacta, nactb);
        return;
    }

    two_body_diff_gen(
        rowa, cola, rowb, colb, V,
        nza, nzb, Xa, Ya, Xb, Yb,
        Vab, XVaXb, XVbXa, IIab,
        nacta, nactb);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H
