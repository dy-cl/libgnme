#include <cassert>
#include <algorithm>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_base.h"
#include "eval/wick_eval.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::spin_overlap(
    arma::umat xhp, arma::umat whp,
    Tc &S, bool alpha)
{
    const size_t &nz = alpha ? m_orba.m_nz : m_orbb.m_nz;
    const size_t &wshift = alpha ? m_orba.m_refx.m_nact : m_orbb.m_refx.m_nact;

    const arma::field<arma::Mat<Tc> > &X = alpha ? m_orba.m_X : m_orbb.m_X;
    const arma::field<arma::Mat<Tc> > &Y = alpha ? m_orba.m_Y : m_orbb.m_Y;

    wick_eval::spin_overlap(xhp, whp, S, nz, X, Y, wshift);

    return;
}

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
