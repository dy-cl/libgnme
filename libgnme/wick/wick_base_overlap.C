#include <cassert>
#include <algorithm>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_base.h"
#include "wick_eval.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::spin_overlap(
    arma::umat xhp, arma::umat whp,
    Tc &S, bool alpha)
{
    // Ensure output is zero'd
    S = Tc(0.0);

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations
    size_t nex = nx + nw;

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_orba.m_nz : m_orbb.m_nz; 

    // Check if the element must be zero
    if(nz > nex) return;

    // Get reference to relevant X/Y matrices for this spin
    const arma::field<arma::Mat<Tc> > &X = alpha ? m_orba.m_X : m_orbb.m_X;
    const arma::field<arma::Mat<Tc> > &Y = alpha ? m_orba.m_Y : m_orbb.m_Y;

    // Shift w indices
    // TODO: Do we want to keep this?
    const size_t &wshift = alpha ? m_orba.m_refx.m_nact : m_orbb.m_refx.m_nact;
    whp += wshift;

    // Get particle-hole indices
    arma::uvec rows, cols;
    wick_eval::indices(xhp, whp, rows, cols);

    if(nex == 0)
    {
        // No excitations, so return simple overlap
        S = (nz == 0) ? Tc(1.0) : Tc(0.0);
    }
    else if(nex == 1)
    {
        // One excitation doesn't require determinant
        S = X(nz)(rows(0),cols(0));
    }
    else
    {
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D;
        wick_eval::build_det(X(0), Y(0), rows, cols, D);

        if(nz == 0)
        {
            S = wick_eval::det(D);
        }
        else
        {
            // Construct matrix with all zero overlaps
            arma::Mat<Tc> Dbar;
            wick_eval::build_det(X(1), Y(1), rows, cols, Dbar);

            if(nz == nex)
            {
                S = wick_eval::det(Dbar);
            }
            else
            {
                // Distribute nz zeros among columns of D.
                // This corresponds to inserting nz columns of Dbar into D for
                // every permutation of the nz zeros.
                arma::Mat<Tc> Dtmp;
                std::vector<size_t> m(nz, 1);
                m.resize(nex, 0);

                do {
                    wick_eval::mix_det(D, Dbar, m, Dtmp);
                    S += wick_eval::det(Dtmp);
                } while(std::prev_permutation(m.begin(), m.end()));
            }
        }
    }

    // Shift w indices
    // TODO: Do we want to keep this?
    whp -= wshift;

    return;
}

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
