// fista_l1_shift_vg.hpp
#pragma once
#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace fista {

    // -------------------------
    // Result / Options
    // -------------------------
    struct Result {
        Eigen::VectorXd x;
        int iters = 0;
        int backtracks_total = 0;
        double smooth_value = std::numeric_limits<double>::quiet_NaN();
        double obj_value = std::numeric_limits<double>::quiet_NaN(); // smooth + beta*|x-m0|_1
        bool converged = false;
    };

    struct Options {
        int max_iters = 500;
        int max_backtracks = 30;

        // Stopping
        double rel_tol = 1e-6;
        double abs_tol = 0.0;

        // Backtracking
        double L0 = 1.0;
        double bt_increase = 2.0;   // L *= bt_increase when condition fails

        // Stability (recommended for nonlinear)
        bool use_monotone = true;   // monotone safeguard
        bool use_fista = true;      // false => ISTA
        bool restart = true;        // restart momentum when harmful

        // Safety clamps
        double L_min = 1e-16;
        double L_max = 1e16;
    };

    // -------------------------
    // Helpers
    // -------------------------

    // soft-thresholding (elementwise)
    // soft(x, tau) = sign(x) * max(|x|-tau, 0)
    inline Eigen::VectorXd soft_threshold(const Eigen::VectorXd& v, double tau) {
        Eigen::VectorXd out(v.size());
        for (int i = 0; i < v.size(); ++i) {
            const double a = std::abs(v[i]);
            if (a <= tau) out[i] = 0.0;
            else out[i] = (v[i] > 0.0 ? 1.0 : -1.0) * (a - tau);
        }
        return out;
    }

    // prox for g(x)=beta*||x-m0||_1 : prox_{t g}(v) = m0 + soft(v-m0, t*beta)
    inline Eigen::VectorXd prox_l1_shift(const Eigen::VectorXd& v,
        const Eigen::VectorXd& m0,
        double t_beta) {
        return m0 + soft_threshold(v - m0, t_beta);
    }

    // ||x-m0||_1
    inline double l1_shift_norm(const Eigen::VectorXd& x, const Eigen::VectorXd& m0) {
        return (x - m0).array().abs().sum();
    }

    // Backtracking condition for smooth f:
    // f(x_new) <= f(y) + grad(y)^T (x_new-y) + (L/2)||x_new-y||^2
    inline bool bt_condition(double f_y,
        const Eigen::VectorXd& grad_y,
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& x_new,
        double f_xnew,
        double L) {
        const Eigen::VectorXd s = x_new - y;
        const double rhs = f_y + grad_y.dot(s) + 0.5 * L * s.squaredNorm();
        return (f_xnew <= rhs);
    }

    // -------------------------
    // Main solver
    // -------------------------
    /**
     * Solve:
     *   minimize  f(x) + beta*||x - m0||_1
     *
     * You provide:
     *   value_grad(x, grad_out) -> returns f(x), writes grad_out = ∇f(x)
     *   value_only(x) -> returns f(x) (optional)
     *
     * If value_only is empty, solver will fall back to calling value_grad and discarding grad.
     *
     * Notes:
     * - Nonlinear F(m) is OK as long as f is differentiable and your grad is consistent.
     * - Backtracking is essential for robustness in nonlinear problems.
     */
    inline Result solve_l1_shifted_fista_vg(
        const Eigen::VectorXd& x_init,
        const Eigen::VectorXd& m0,
        double beta,
        const std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>& value_grad,
        const std::function<double(const Eigen::VectorXd&)>& value_only, // may be empty {}
        const Options& opt = Options()
    ) {
        if (x_init.size() != m0.size()) throw std::runtime_error("x_init and m0 size mismatch.");
        if (!(beta >= 0.0)) throw std::runtime_error("beta must be >= 0.");
        if (opt.L0 <= 0.0) throw std::runtime_error("L0 must be > 0.");
        if (!value_grad) throw std::runtime_error("value_grad callback is empty.");

        Result res;

        Eigen::VectorXd x = x_init;
        Eigen::VectorXd y = x_init;

        Eigen::VectorXd grad_y(x.size());

        double t = 1.0; // Nesterov parameter
        double L = std::min(std::max(opt.L0, opt.L_min), opt.L_max);

        // Monotone tracking
        double best_obj = std::numeric_limits<double>::infinity();
        Eigen::VectorXd best_x = x;

        auto eval_value_only = [&](const Eigen::VectorXd& m) -> double {
            if (value_only) return value_only(m);
            // fallback: compute via value_grad and discard grad
            Eigen::VectorXd tmpg(m.size());
            return value_grad(m, tmpg);
            };

        for (int k = 0; k < opt.max_iters; ++k) {
            // 1) compute f(y) and ∇f(y) in ONE call
            const double f_y = value_grad(y, grad_y);

            // 2) backtracking to find adequate L
            Eigen::VectorXd x_new(x.size());
            double f_xnew = 0.0;
            int bt = 0;

            for (; bt < opt.max_backtracks; ++bt) {
                const double step = 1.0 / L;

                // gradient step
                const Eigen::VectorXd v = y - step * grad_y;

                // proximal step (L1 shift)
                x_new = prox_l1_shift(v, m0, step * beta);

                // smooth value only (cheap if you skip grad inside Optimize)
                f_xnew = eval_value_only(x_new);

                // check sufficient decrease upper bound
                if (bt_condition(f_y, grad_y, y, x_new, f_xnew, L)) break;

                L *= opt.bt_increase;
                if (L > opt.L_max) break;
            }

            res.backtracks_total += bt;

            // 3) objective value (for monotone safeguard)
            double obj_new = f_xnew + beta * l1_shift_norm(x_new, m0);

            if (opt.use_monotone) {
                if (obj_new < best_obj) {
                    best_obj = obj_new;
                    best_x = x_new;
                }
                else {
                    // snap back to best solution so far (monotone behavior)
                    x_new = best_x;
                    f_xnew = eval_value_only(x_new);
                    obj_new = f_xnew + beta * l1_shift_norm(x_new, m0);
                }
            }

            // 4) stopping by relative step
            const double step_norm = (x_new - x).norm();
            const double denom = std::max(1.0, x.norm());
            if (step_norm <= opt.abs_tol || (step_norm / denom) <= opt.rel_tol) {
                x = x_new;
                res.converged = true;
                res.iters = k + 1;
                res.x = x;
                res.smooth_value = eval_value_only(x);
                res.obj_value = res.smooth_value + beta * l1_shift_norm(x, m0);
                return res;
            }

            // 5) acceleration
            Eigen::VectorXd y_new = x_new;

            if (opt.use_fista) {
                const double t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
                const double momentum = (t - 1.0) / t_new;

                // y_{k+1} = x_{k+1} + momentum*(x_{k+1}-x_k)
                y_new = x_new + momentum * (x_new - x);

                if (opt.restart) {
                    // restart if momentum seems harmful:
                    // (x_new - x) · (y_new - x_new) > 0
                    const Eigen::VectorXd s = x_new - x;
                    const Eigen::VectorXd p = y_new - x_new;
                    if (s.dot(p) > 0.0) {
                        y_new = x_new;
                        t = 1.0;
                    }
                    else {
                        t = t_new;
                    }
                }
                else {
                    t = t_new;
                }
            }
            else {
                // ISTA
                y_new = x_new;
                t = 1.0;
            }

            // update iterates
            x = x_new;
            y = y_new;

            res.iters = k + 1;
        }

        // max iters reached
        res.x = x;
        res.smooth_value = eval_value_only(x);
        res.obj_value = res.smooth_value + beta * l1_shift_norm(x, m0);
        res.converged = false;
        return res;
    }

} // namespace fista
