#include <cassert>
#include <algorithm>
#include <libgnme/utils/eri_ao2mo.h>
#include <libgnme/utils/linalg.h>
#include "eval/wick_eval.h"
#include "wick_base.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::same_spin_two_body(
    Tc &V, bool alpha,
    wick_eval::same_scratch<Tc> &work)
{
    const size_t &nz = alpha ? m_orba.m_nz : m_orbb.m_nz;
    const arma::Col<Tc> &V0 = alpha ? m_two_body_int->Vaa : m_two_body_int->Vbb;
    const arma::field<arma::Mat<Tc> > &XVX = alpha ? m_two_body_int->XVaXa : m_two_body_int->XVbXb;
    arma::field<arma::Mat<Tc> > &II = alpha ? m_two_body_int->IIaa : m_two_body_int->IIbb;

    const size_t nactx = alpha ? m_orba.m_refx.m_nact : m_orbb.m_refx.m_nact;
    const size_t nactw = alpha ? m_orba.m_refw.m_nact : m_orbb.m_refw.m_nact;
    const size_t nact = nactx + nactw;

    wick_eval::two_body_same(V, nz, V0, XVX, II, nact, work);
}

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::diff_spin_two_body(
    Tc &V,
    wick_eval::same_scratch<Tc> &awork,
    wick_eval::same_scratch<Tc> &bwork,
    wick_eval::diff_scratch<Tc> &work)
{
    const size_t &nza = m_orba.m_nz;
    const size_t &nzb = m_orbb.m_nz;

    const arma::Mat<Tc> &Vab = m_two_body_int->Vab;
    const arma::field<arma::Mat<Tc> > &XVaXb = m_two_body_int->XVaXb;
    const arma::field<arma::Mat<Tc> > &XVbXa = m_two_body_int->XVbXa;
    arma::field<arma::Mat<Tc> > &IIab = m_two_body_int->IIab;

    const size_t nacta = m_orba.m_refx.m_nact + m_orba.m_refw.m_nact;
    const size_t nactb = m_orbb.m_refx.m_nact + m_orbb.m_refw.m_nact;

    wick_eval::two_body_diff(
        V, nza, nzb,
        Vab, XVaXb, XVbXa, IIab,
        nacta, nactb,
        awork, bwork, work);
}

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
