#include <cassert>
#include <algorithm>
#include <libgnme/utils/linalg.h>
#include "wick_base.h"
#include "eval/wick_eval.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::spin_one_body(
    Tc &F, bool alpha,
    wick_eval::same_scratch<Tc> &work)
{
    const size_t &nz = alpha ? m_orba.m_nz : m_orbb.m_nz;
    const arma::Col<Tc> &F0 = alpha ? m_one_body_int->F0a : m_one_body_int->F0b;
    const arma::field<arma::Mat<Tc> > &XFX = alpha ? m_one_body_int->XFXa : m_one_body_int->XFXb;

    wick_eval::spin_one_body(F, nz, F0, XFX, work);
}

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
