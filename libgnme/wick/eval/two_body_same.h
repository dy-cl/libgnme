#ifndef LIBGNME_WICK_EVAL_TWO_BODY_SAME_H
#define LIBGNME_WICK_EVAL_TWO_BODY_SAME_H

#include <armadillo>
#include <cstdint>
#include <libgnme/utils/linalg.h>

#include "helpers.h"
#include "prepare.h"

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

/** \brief Evaluate same-spin two-body matrix element for nz = 0 and zero excitations.
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

/** \brief Evaluate same-spin two-body matrix element for nz = 0 and one excitation.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param X Lower-triangular contractions.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_m0_l1(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX)
{
    const size_t r0 = rows(0);
    const size_t c0 = cols(0);

    V = V0(0) * X(0)(r0,c0) - Tc(2.0) * XVX(0,0,0)(r0,c0);
}

/** \brief Evaluate same-spin two-body matrix element for nz = 0 and two excitations.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nact Total active dimension for flattened two-electron intermediates.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_m0_l2(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact)
{
    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    const Tc a00 = D(0,0);
    const Tc a01 = D(0,1);
    const Tc a10 = D(1,0);
    const Tc a11 = D(1,1);
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
}

/** \brief Evaluate same-spin two-body matrix element for nz = 0 and arbitrary excitation rank.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nact Total active dimension for flattened two-electron intermediates.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_m0_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact)
{
    const size_t nex = rows.n_elem;

    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    Tc detD;
    size_t nzero;
    arma::Mat<Tc> cofD;
    adjoint_matrix(D, cofD, detD, nzero);
    cofD = cofD.t();

    V = V0(0) * detD;

    const arma::Mat<Tc> &JK = XVX(0,0,0);

    for(size_t k=0; k<nex; k++)
    {
        const size_t ck = cols(k);

        const Tc corr = column_replacement_correction(
            D, cofD, k,
            [&](const size_t r) {
                return JK(rows(r), ck);
            });

        V -= Tc(2.0) * (detD + corr);
    }

    arma::Mat<Tc> Dminor;
    arma::Mat<Tc> cof_minor;

    for(size_t i=0; i<nex; i++)
    for(size_t j=0; j<nex; j++)
    {
        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;
        const size_t r_fixed = rows(i);
        const size_t c_fixed = cols(j);

        minor_adjt(D, i, j, Dminor, cof_minor,
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
                            i, j, r, k, r_fixed, c_fixed);
                        });

                    V += Tc(0.5 * phase) * (det_minor_val + corr);
                }
            });
    }
}

/** \brief Dispatch same-spin two-body matrix element for nz = 0.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nact Total active dimension for flattened two-electron intermediates.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_m0(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact)
{
    const size_t nex = rows.n_elem;

    if(nex == 0)
    {
        two_body_same_m0_l0(V, V0);
        return;
    }

    if(nex == 1)
    {
        two_body_same_m0_l1(rows, cols, V, X, V0, XVX);
        return;
    }

    if(nex == 2)
    {
        two_body_same_m0_l2(rows, cols, V, X, Y, V0, XVX, II, nact);
        return;
    }

    two_body_same_m0_gen(rows, cols, V, X, Y, V0, XVX, II, nact);
}

/** \brief Evaluate same-spin two-body matrix element for the generic nz > 0 case.
    \tparam Tc Matrix element type.
    \param rows Row indices.
    \param cols Column indices.
    \param V Output two-body matrix element.
    \param nz Number of zero-overlap orbital pairs.
    \param X Lower-triangular contractions.
    \param Y Upper-triangular contractions.
    \param V0 Zeroth-order same-spin two-body contractions.
    \param XVX First-order same-spin two-body contractions.
    \param II Same-spin two-electron intermediates.
    \param nact Total active dimension for flattened two-electron intermediates.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline void two_body_same_gen(
    const arma::uvec &rows, const arma::uvec &cols,
    Tc &V,
    const size_t &nz,
    const arma::field<arma::Mat<Tc> > &X,
    const arma::field<arma::Mat<Tc> > &Y,
    const arma::Col<Tc> &V0,
    const arma::field<arma::Mat<Tc> > &XVX,
    arma::field<arma::Mat<Tc> > &II,
    const size_t nact)
{
    const size_t nex = rows.n_elem;
    const size_t d = (nz > 0) ? 2 : 1;

    arma::Mat<Tc> D;
    build_det(X(0), Y(0), rows, cols, D);

    arma::Mat<Tc> Db;
    build_det(X(1), Y(1), rows, cols, Db);

    arma::field<arma::Mat<Tc> > JKtmp(d,d,d);
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
    for(size_t k=0; k<d; k++)
        build_mat(XVX(i,j,k), rows, cols, JKtmp(i,j,k));

    arma::Mat<Tc> Dtmp;

    for_each_m_combination(nex+2, nz, [&](uint64_t bits) {
        const size_t m0 = bit(bits, 0);
        const size_t m1 = bit(bits, 1);

        mix_det(D, Db, bits, 2, Dtmp);

        Tc detDtmp;
        size_t nzero;
        arma::Mat<Tc> adjDtmp;
        adjoint_matrix(Dtmp, adjDtmp, detDtmp, nzero);
        adjDtmp = adjDtmp.t();

        V += V0(m0 + m1) * detDtmp;

        for(size_t i=0; i<nex; i++)
        {
            const size_t mi = bit(bits, i+2);

            arma::Col<Tc> v1(JKtmp(m0,m1,mi).colptr(i), nex, false, true);
            arma::Col<Tc> v2(Dtmp.colptr(i), nex, false, true);
            arma::Col<Tc> a(adjDtmp.colptr(i), nex, false, true);

            V -= Tc(2.0) * (detDtmp + arma::dot(v1-v2, a));
        }

        arma::field<arma::Mat<Tc> > IItmp(d);
        arma::Mat<Tc> D2, Db2, Dtmp2;

        for(size_t i=0; i<nex; i++)
        for(size_t j=0; j<nex; j++)
        {
            for(size_t x=0; x<d; x++)
            {
                IItmp(x).set_size(nex-1, nex-1);

                for(size_t kk=0; kk<nex-1; kk++)
                for(size_t rr=0; rr<nex-1; rr++)
                {
                    const size_t r_full = minor_to_full(rr, i);
                    const size_t k_full = minor_to_full(kk, j);

                    IItmp(x)(rr,kk) = two_body_same_ii(
                        II, nact,
                        m0, m1, bit(bits, 2), x,
                        rows(i), cols(j),
                        rows(r_full), cols(k_full));
                }
            }

            D2 = D;
            D2.shed_row(i);
            D2.shed_col(j);

            Db2 = Db;
            Db2.shed_row(i);
            Db2.shed_col(j);

            mix_det(D2, Db2, bits, 3, Dtmp2);

            Tc detDtmp2;
            size_t nzero2;
            arma::Mat<Tc> adjDtmp2;
            adjoint_matrix(Dtmp2, adjDtmp2, detDtmp2, nzero2);
            adjDtmp2 = adjDtmp2.t();

            const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;

            for(size_t k=0; k<nex-1; k++)
            {
                const size_t mk = bit(bits, k+3);

                arma::Col<Tc> v1(IItmp(mk).colptr(k), nex-1, false, true);
                arma::Col<Tc> v2(Dtmp2.colptr(k), nex-1, false, true);
                arma::Col<Tc> a(adjDtmp2.colptr(k), nex-1, false, true);

                V += Tc(0.5 * phase) * (detDtmp2 + arma::dot(v1-v2, a));
            }
        }
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

    whp += nactx;

    arma::uvec rows, cols;
    indices(xhp, whp, rows, cols);

    const size_t nact = nactx + nactw;

    if(nz == 0)
    {
        two_body_same_m0(rows, cols, V, X, Y, V0, XVX, II, nact);
        return;
    }

    two_body_same_gen(rows, cols, V, nz, X, Y, V0, XVX, II, nact);
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_TWO_BODY_SAME_H
