#ifndef LIBGNME_WICK_EVAL_SCRATCH_H
#define LIBGNME_WICK_EVAL_SCRATCH_H

#include <armadillo>

namespace libgnme {
namespace wick_eval {

/** \brief Resize a scratch vector.
    \tparam T Element type.
    \param x Scratch vector.
    \param n Required size.
    \ingroup gnme_wick
 **/
template<typename T>
inline void ensure_vec(arma::Col<T> &x, const size_t n)
{
    if(x.n_elem != n) x.set_size(n);
}

/** \brief Resize a scratch matrix.
    \tparam T Element type.
    \param x Scratch matrix.
    \param n Required row count.
    \param m Required column count.
    \ingroup gnme_wick
 **/
template<typename T>
inline void ensure_mat(arma::Mat<T> &x, const size_t n, const size_t m)
{
    if(x.n_rows != n || x.n_cols != m) x.set_size(n,m);
}

/** \brief Same-spin Wick scratch storage.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
struct same_scratch
{
    arma::uvec rows;
    arma::uvec cols;

    arma::Mat<Tc> det0;
    arma::Mat<Tc> det1;
    arma::Mat<Tc> det_mix;

    arma::Col<Tc> fcol;
    arma::Col<Tc> dv;
    arma::Col<Tc> v1;
    arma::Col<Tc> dv1;
    arma::Col<Tc> dv1m;

    arma::Mat<Tc> jslice_full;
    arma::Mat<Tc> jslice2;
    arma::Mat<Tc> det_mix2;

    arma::Mat<Tc> adjt_det;
    arma::Mat<Tc> adjt_det2;

    arma::Col<Tc> invs;
    arma::Col<Tc> invslm1;

    arma::Mat<Tc> lu;

    void ensure(const size_t l)
    {
        if(rows.n_elem != l) rows.set_size(l);
        if(cols.n_elem != l) cols.set_size(l);

        ensure_mat(det0, l, l);
        ensure_mat(det1, l, l);
        ensure_mat(det_mix, l, l);
        ensure_mat(adjt_det, l, l);
        ensure_mat(jslice_full, l, l);

        ensure_vec(fcol, l);
        ensure_vec(dv, l);
        ensure_vec(v1, l);
        ensure_vec(dv1, l);
        ensure_vec(invs, l);

        ensure_mat(lu, 6, 6);

        const size_t lm1 = l ? l - 1 : 0;

        ensure_vec(dv1m, lm1);
        ensure_vec(invslm1, lm1);
        ensure_mat(det_mix2, lm1, lm1);
        ensure_mat(jslice2, lm1, lm1);
        ensure_mat(adjt_det2, lm1, lm1);
    }
};

/** \brief Different-spin Wick scratch storage.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
struct diff_scratch
{
    arma::Mat<Tc> deta0;
    arma::Mat<Tc> deta1;
    arma::Mat<Tc> deta_mix;
    arma::Mat<Tc> detb0;
    arma::Mat<Tc> detb1;
    arma::Mat<Tc> detb_mix;

    arma::Mat<Tc> deta_mix_minor;
    arma::Mat<Tc> detb_mix_minor;

    arma::Mat<Tc> adjt_deta;
    arma::Mat<Tc> adjt_detb;
    arma::Mat<Tc> adjt_deta_mix_minor;
    arma::Mat<Tc> adjt_detb_mix_minor;

    arma::Col<Tc> v1a;
    arma::Col<Tc> v1b;
    arma::Col<Tc> dv1a;
    arma::Col<Tc> dv1b;

    arma::Mat<Tc> iislicea;
    arma::Mat<Tc> iisliceb;

    arma::Col<Tc> invsla;
    arma::Col<Tc> invslb;
    arma::Col<Tc> invslam1;
    arma::Col<Tc> invslbm1;

    arma::Mat<Tc> lua;
    arma::Mat<Tc> lub;

    void ensure(const size_t la, const size_t lb)
    {
        ensure_mat(deta0, la, la);
        ensure_mat(deta1, la, la);
        ensure_mat(deta_mix, la, la);
        ensure_mat(adjt_deta, la, la);
        ensure_vec(v1a, la);
        ensure_vec(dv1a, la);
        ensure_mat(iislicea, la, la);
        ensure_vec(invsla, la);
        ensure_mat(lua, 6, 6);

        const size_t lam1 = la ? la - 1 : 0;
        ensure_mat(deta_mix_minor, lam1, lam1);
        ensure_mat(adjt_deta_mix_minor, lam1, lam1);
        ensure_vec(invslam1, lam1);

        ensure_mat(detb0, lb, lb);
        ensure_mat(detb1, lb, lb);
        ensure_mat(detb_mix, lb, lb);
        ensure_mat(adjt_detb, lb, lb);
        ensure_vec(v1b, lb);
        ensure_vec(dv1b, lb);
        ensure_mat(iisliceb, lb, lb);
        ensure_vec(invslb, lb);
        ensure_mat(lub, 6, 6);

        const size_t lbm1 = lb ? lb - 1 : 0;
        ensure_mat(detb_mix_minor, lbm1, lbm1);
        ensure_mat(adjt_detb_mix_minor, lbm1, lbm1);
        ensure_vec(invslbm1, lbm1);
    }
};

/** \brief Full Wick scratch storage for alpha, beta, and different-spin paths.
    \tparam Tc Matrix element type.
    \ingroup gnme_wick
 **/
template<typename Tc>
struct scratch
{
    same_scratch<Tc> aa;
    same_scratch<Tc> bb;
    diff_scratch<Tc> diff;
};

/** \brief Return thread-local Wick evaluator scratch storage.
    \tparam Tc Matrix element type.
    \return Thread-local scratch storage.
    \ingroup gnme_wick
 **/
template<typename Tc>
inline scratch<Tc> &local_scratch()
{
    static thread_local scratch<Tc> work;
    return work;
}

} // namespace wick_eval
} // namespace libgnme

#endif // LIBGNME_WICK_EVAL_SCRATCH_H
