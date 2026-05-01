#include <cassert>
#include <algorithm>
#include <libgnme/utils/linalg.h>
#include "wick_base.h"
#include "eval/wick_eval.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::spin_one_body(
    arma::umat xhp, arma::umat whp,
    Tc &F, bool alpha)
{
    const size_t &nz = alpha ? m_orba.m_nz : m_orbb.m_nz;
    const size_t &wshift = alpha ? m_orba.m_refx.m_nact : m_orbb.m_refx.m_nact;

    const arma::field<arma::Mat<Tc> > &X = alpha ? m_orba.m_X : m_orbb.m_X;
    const arma::field<arma::Mat<Tc> > &Y = alpha ? m_orba.m_Y : m_orbb.m_Y;

    const arma::Col<Tc> &F0 = alpha ? m_one_body_int->F0a : m_one_body_int->F0b;
    const arma::field<arma::Mat<Tc> > &XFX = alpha ? m_one_body_int->XFXa : m_one_body_int->XFXb;

    wick_eval::spin_one_body(xhp, whp, F, nz, X, Y, F0, XFX, wshift);

    return;
}

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
