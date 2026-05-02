#ifndef LIBGNME_WICK_EVAL_SCRATCH_H
#define LIBGNME_WICK_EVAL_SCRATCH_H

#include <armadillo>
#include <cassert>

namespace libgnme {
namespace wick_eval {

template<typename T>
inline void reserve_vec(arma::Col<T> &x, const size_t n)
{
    if(x.n_elem < n) x.set_size(n);
}

template<typename T>
inline void reserve_mat(arma::Mat<T> &x, const size_t n, const size_t m)
{
    if(x.n_rows < n || x.n_cols < m) x.set_size(n, m);
}

inline void reserve_uvec(arma::uvec &x, const size_t n)
{
    if(x.n_elem < n) x.set_size(n);
}

template<typename Tc>
struct same_scratch
{
    size_t l = 0;
    size_t lcap = 0;

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

    void ensure_capacity(const size_t max_l)
    {
        if(lcap >= max_l) return;

        lcap = max_l;

        reserve_uvec(rows, max_l);
        reserve_uvec(cols, max_l);

        reserve_mat(det0, max_l, max_l);
        reserve_mat(det1, max_l, max_l);
        reserve_mat(det_mix, max_l, max_l);
        reserve_mat(adjt_det, max_l, max_l);
        reserve_mat(jslice_full, max_l, max_l);

        reserve_vec(fcol, max_l);
        reserve_vec(dv, max_l);
        reserve_vec(v1, max_l);
        reserve_vec(dv1, max_l);
        reserve_vec(invs, max_l);

        const size_t max_lm1 = max_l ? max_l - 1 : 0;
        reserve_vec(dv1m, max_lm1);
        reserve_vec(invslm1, max_lm1);

        reserve_mat(det_mix2, max_lm1, max_lm1);
        reserve_mat(jslice2, max_lm1, max_lm1);
        reserve_mat(adjt_det2, max_lm1, max_lm1);

        reserve_mat(lu, 6, 6);
    }

    void set_active(const size_t active_l)
    {
        assert(active_l <= lcap);
        l = active_l;
    }
};

template<typename Tc>
struct diff_scratch
{
    size_t la = 0;
    size_t lb = 0;
    size_t lacap = 0;
    size_t lbcap = 0;

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

    void ensure_capacity(const size_t max_la, const size_t max_lb)
    {
        if(lacap < max_la) {
            lacap = max_la;

            reserve_mat(deta0, max_la, max_la);
            reserve_mat(deta1, max_la, max_la);
            reserve_mat(deta_mix, max_la, max_la);
            reserve_mat(adjt_deta, max_la, max_la);
            reserve_mat(iislicea, max_la, max_la);

            reserve_vec(v1a, max_la);
            reserve_vec(dv1a, max_la);
            reserve_vec(invsla, max_la);

            const size_t max_lam1 = max_la ? max_la - 1 : 0;
            reserve_mat(deta_mix_minor, max_lam1, max_lam1);
            reserve_mat(adjt_deta_mix_minor, max_lam1, max_lam1);
            reserve_vec(invslam1, max_lam1);

            reserve_mat(lua, 6, 6);
        }

        if(lbcap < max_lb) {
            lbcap = max_lb;

            reserve_mat(detb0, max_lb, max_lb);
            reserve_mat(detb1, max_lb, max_lb);
            reserve_mat(detb_mix, max_lb, max_lb);
            reserve_mat(adjt_detb, max_lb, max_lb);
            reserve_mat(iisliceb, max_lb, max_lb);

            reserve_vec(v1b, max_lb);
            reserve_vec(dv1b, max_lb);
            reserve_vec(invslb, max_lb);

            const size_t max_lbm1 = max_lb ? max_lb - 1 : 0;
            reserve_mat(detb_mix_minor, max_lbm1, max_lbm1);
            reserve_mat(adjt_detb_mix_minor, max_lbm1, max_lbm1);
            reserve_vec(invslbm1, max_lbm1);

            reserve_mat(lub, 6, 6);
        }
    }

    void set_active(const size_t active_la, const size_t active_lb)
    {
        assert(active_la <= lacap);
        assert(active_lb <= lbcap);
        la = active_la;
        lb = active_lb;
    }
};

template<typename Tc>
struct scratch
{
    same_scratch<Tc> aa;
    same_scratch<Tc> bb;
    diff_scratch<Tc> diff;

    void ensure_capacity(const size_t max_la, const size_t max_lb)
    {
        aa.ensure_capacity(max_la);
        bb.ensure_capacity(max_lb);
        diff.ensure_capacity(max_la, max_lb);
    }
};

template<typename Tc>
inline scratch<Tc> &local_scratch()
{
    static thread_local scratch<Tc> work;
    return work;
}

} // namespace wick_eval
} // namespace libgnme

#endif
