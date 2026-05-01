#ifndef LIBGNME_WICK_EVAL_TWO_BODY_SAME_H
#define LIBGNME_WICK_EVAL_TWO_BODY_SAME_H

#include <armadillo>
#include <cstdint>
#include <libgnme/utils/linalg.h>

#include "helpers.h"
#include "prepare.h"

namespace libgnme {
namespace wick_eval {

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

    arma::Mat<Tc> JK;
    build_mat(XVX(0,0,0), rows, cols, JK);

    Tc detD;
    size_t nzero;
    arma::Mat<Tc> adjD;
    adjoint_matrix(D, adjD, detD, nzero);
    adjD = adjD.t();

    V = V0(0) * detD;

    for(size_t i=0; i<nex; i++)
    {
        arma::Col<Tc> v1(JK.colptr(i), nex, false, true);
        arma::Col<Tc> v2(D.colptr(i), nex, false, true);
        arma::Col<Tc> a(adjD.colptr(i), nex, false, true);

        V -= Tc(2.0) * (detD + arma::dot(v1-v2, a));
    }

    arma::field<arma::Mat<Tc> > IItmp(1);
    arma::Mat<Tc> Dtmp2;

    for(size_t i=0; i<nex; i++)
    for(size_t j=0; j<nex; j++)
    {
        arma::Mat<Tc> vIItmp(
            II(0,0).colptr(nact * rows(i) + cols(j)),
            nact, nact, false, true);

        IItmp(0) = vIItmp.submat(cols, rows).st();
        IItmp(0).shed_row(i);
        IItmp(0).shed_col(j);

        Dtmp2 = D;
        Dtmp2.shed_row(i);
        Dtmp2.shed_col(j);

        Tc detDtmp2;
        size_t nzero2;
        arma::Mat<Tc> adjDtmp2;
        adjoint_matrix(Dtmp2, adjDtmp2, detDtmp2, nzero2);
        adjDtmp2 = adjDtmp2.t();

        const double phase = ((i % 2) xor (j % 2)) ? -1.0 : 1.0;

        for(size_t k=0; k<nex-1; k++)
        {
            arma::Col<Tc> v1(IItmp(0).colptr(k), nex-1, false, true);
            arma::Col<Tc> v2(Dtmp2.colptr(k), nex-1, false, true);
            arma::Col<Tc> a(adjDtmp2.colptr(k), nex-1, false, true);

            V += Tc(0.5 * phase) * (detDtmp2 + arma::dot(v1-v2, a));
        }
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
                arma::Mat<Tc> vIItmp(
                    II(2 * bit(bits, 2) + x, 2 * m0 + m1).colptr(nact * rows(i) + cols(j)),
                    nact, nact, false, true);

                IItmp(x) = vIItmp.submat(cols, rows).st();
                IItmp(x).shed_row(i);
                IItmp(x).shed_col(j);
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
