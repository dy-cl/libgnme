#include <cassert>
#include <algorithm>
#include <libgnme/utils/eri_ao2mo.h>
#include <libgnme/utils/linalg.h>
#include "eval/wick_eval.h"
#include "wick_base.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::same_spin_two_body(
    arma::umat xhp, arma::umat whp,
    Tc &V, bool alpha)
{
    const size_t &nz = alpha ? m_orba.m_nz : m_orbb.m_nz;

    const arma::field<arma::Mat<Tc> > &X = alpha ? m_orba.m_X : m_orbb.m_X;
    const arma::field<arma::Mat<Tc> > &Y = alpha ? m_orba.m_Y : m_orbb.m_Y;

    const arma::Col<Tc> &V0 = alpha ? m_two_body_int->Vaa : m_two_body_int->Vbb;
    const arma::field<arma::Mat<Tc> > &XVX = alpha ? m_two_body_int->XVaXa : m_two_body_int->XVbXb;

    arma::field<arma::Mat<Tc> > &II = alpha ? m_two_body_int->IIaa : m_two_body_int->IIbb;

    const size_t nactx = alpha ? m_orba.m_refx.m_nact : m_orbb.m_refx.m_nact;
    const size_t nactw = alpha ? m_orba.m_refw.m_nact : m_orbb.m_refw.m_nact;

    wick_eval::two_body_same(xhp, whp, V, nz, X, Y, V0, XVX, II, nactx, nactw);

    return;
}

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::diff_spin_two_body(
    arma::umat xahp, arma::umat xbhp,
    arma::umat wahp, arma::umat wbhp,
    Tc &V)
{
    const size_t &nza = m_orba.m_nz;
    const size_t &nzb = m_orbb.m_nz;

    const arma::field<arma::Mat<Tc> > &Xa = m_orba.m_X;
    const arma::field<arma::Mat<Tc> > &Ya = m_orba.m_Y;
    const arma::field<arma::Mat<Tc> > &Xb = m_orbb.m_X;
    const arma::field<arma::Mat<Tc> > &Yb = m_orbb.m_Y;

    const arma::Mat<Tc> &Vab = m_two_body_int->Vab;
    const arma::field<arma::Mat<Tc> > &XVaXb = m_two_body_int->XVaXb;
    const arma::field<arma::Mat<Tc> > &XVbXa = m_two_body_int->XVbXa;

    arma::field<arma::Mat<Tc> > &IIab = m_two_body_int->IIab;
    arma::field<arma::Mat<Tc> > &IIba = m_two_body_int->IIba;

    const size_t nactxa = m_orba.m_refx.m_nact;
    const size_t nactwa = m_orba.m_refw.m_nact;
    const size_t nactxb = m_orbb.m_refx.m_nact;
    const size_t nactwb = m_orbb.m_refw.m_nact;

    wick_eval::two_body_diff(
    xahp, xbhp, wahp, wbhp, V,
    nza, nzb, Xa, Ya, Xb, Yb,
    Vab, XVaXb, XVbXa, IIab,
    nactxa, nactwa, nactxb, nactwb);

    return;
}

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
