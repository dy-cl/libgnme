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

/** \brief Build a mixed determinant minor and return its determinant.
    \tparam Tc Matrix element type.
    \param D Non-zero-overlap branch determinant.
    \param Db Zero-overlap branch determinant.
    \param bits Zero-distribution bitstring.
    \param offset Offset into bitstring for determinant columns.
    \param row_rm Removed row.
    \param col_rm Removed column.
    \param Dminor Scratch determinant minor.
    \return Determinant of the mixed minor.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline Tc mixed_minor_det(
    const arma::Mat<Tc> &D,
    const arma::Mat<Tc> &Db,
    const uint64_t bits,
    const size_t offset,
    const size_t row_rm,
    const size_t col_rm,
    arma::Mat<Tc> &Dminor)
{
    const size_t l = D.n_rows;
    const size_t lm1 = l - 1;

    Dminor.set_size(lm1,lm1);

    for(size_t j=0; j<lm1; j++)
    for(size_t i=0; i<lm1; i++)
    {
        const size_t fi = minor_to_full(i,row_rm);
        const size_t fj = minor_to_full(j,col_rm);
        const arma::Mat<Tc> &src = bit(bits,j+offset) ? Db : D;

        Dminor(i,j) = src(fi,fj);
    }

    return det(Dminor);
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

    two_body_diff_gen(
        rowa, cola, rowb, colb, V,
        nza, nzb, Xa, Ya, Xb, Yb,
        Vab, XVaXb, XVbXa, IIab,
        nacta, nactb);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_TWO_BODY_DIFF_H
