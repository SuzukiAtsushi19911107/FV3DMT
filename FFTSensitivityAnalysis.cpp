/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once 
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#pragma warning(disable : 4996)
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include "FFTSensitivityAnalysis.h"
#include <fstream>
#include "optim.hpp"
#include <ostream>
#include "Analysis.h"
#include <boost/math/distributions/chi_squared.hpp>
using boost::math::chi_squared;
using boost::math::quantile;
#define strcasecmp _stricmp
// mt_sensitivity.cpp
// C++17 / Eigen only (FFT: unsupported/Eigen/FFT)

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <limits>
#include <stdexcept>
#include <complex>
#include <cstdint>
#include <unsupported/Eigen/FFT>
#include <Eigen/Dense>
#include <iomanip>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace {
    // ========================= ユーティリティ =========================
    static void write_D_cols(
        const std::string& filename,
        const std::vector<std::vector<double>>& D_cols)
    {
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        const int k = (int)D_cols.size();
        const int n = (int)D_cols.front().size();

        ofs << "# nModel " << n << "\n";
        ofs << "# nDirection " << k << "\n";

        // 各セルごとに出す（1行 = 1セル）
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                ofs << D_cols[j][i];
                if (j != k - 1) ofs << " ";
            }
            ofs << "\n";
        }

        ofs.close();
    }
    static inline int _sym_idx(int i, int N, int c) { return (2 * c - i + 2 * N) % N; }


    static inline double max_abs_value(const std::vector<double>& v) {
        double m = 0.0;
        for (double x : v) {
            m = std::max(m, std::abs(x));
        }
        return m;
    }

    static inline double l2norm(const std::vector<double>& v) {
        long long n = v.size(); long double s = 0;
        for (long long i = 0; i < n; ++i) { long double t = v[i]; s += t * t; }
        return std::sqrt((double)s);
    }
    static inline void normalize_inplace(std::vector<double>& v, double eps = 1e-12) {
        double n = l2norm(v);
        if (n >= eps) { for (double& x : v) x /= n; }
    }

    struct GridSpec {
        int nx, ny, nz;  // サイズ
        double ox, oy, oz, dx, dy, dz; // 原点・間隔
    };
    static inline size_t idx3(int i, int j, int k, int nx, int ny) { return (size_t)(k * ny + j) * nx + i; }

    // ========================= 3D FFTヘルパ =========================
    // 3D配列を std::vector<std::complex<double>> で保持
    // x,y,z 各軸に沿って 1D FFT を適用
    struct FFT3D {
        int nx, ny, nz;
        Eigen::FFT<double> fft;

        FFT3D(int nx_, int ny_, int nz_) :nx(nx_), ny(ny_), nz(nz_) {}

        // 前方向
        void fwd(std::vector<std::complex<double>>& out,
            const std::vector<std::complex<double>>& in) {
            out = in;
            // x方向
            std::vector<std::complex<double>> line(nx), outLine(nx);
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) line[i] = out[idx3(i, j, k, nx, ny)];
                    fft.fwd(outLine, line);
                    for (int i = 0; i < nx; ++i) out[idx3(i, j, k, nx, ny)] = outLine[i];
                }
            // y方向
            line.resize(ny); outLine.resize(ny);
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i) {
                    for (int j = 0; j < ny; ++j) line[j] = out[idx3(i, j, k, nx, ny)];
                    fft.fwd(outLine, line);
                    for (int j = 0; j < ny; ++j) out[idx3(i, j, k, nx, ny)] = outLine[j];
                }
            // z方向
            line.resize(nz); outLine.resize(nz);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    for (int k = 0; k < nz; ++k) line[k] = out[idx3(i, j, k, nx, ny)];
                    fft.fwd(outLine, line);
                    for (int k = 0; k < nz; ++k) out[idx3(i, j, k, nx, ny)] = outLine[k];
                }
        }

        // 逆変換（Eigen::FFT は逆で自動1/N 正規化）
        void inv(std::vector<std::complex<double>>& out,
            const std::vector<std::complex<double>>& in) {
            out = in;
            std::vector<std::complex<double>> line, outLine;
            // 逆z
            line.resize(nz); outLine.resize(nz);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    for (int k = 0; k < nz; ++k) line[k] = out[idx3(i, j, k, nx, ny)];
                    fft.inv(outLine, line);
                    for (int k = 0; k < nz; ++k) out[idx3(i, j, k, nx, ny)] = outLine[k];
                }
            // 逆y
            line.resize(ny); outLine.resize(ny);
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i) {
                    for (int j = 0; j < ny; ++j) line[j] = out[idx3(i, j, k, nx, ny)];
                    fft.inv(outLine, line);
                    for (int j = 0; j < ny; ++j) out[idx3(i, j, k, nx, ny)] = outLine[j];
                }
            // 逆x
            line.resize(nx); outLine.resize(nx);
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) line[i] = out[idx3(i, j, k, nx, ny)];
                    fft.inv(outLine, line);
                    for (int i = 0; i < nx; ++i) out[idx3(i, j, k, nx, ny)] = outLine[i];
                }
        }
    };

    // ========================= fftshift / ifftshift =========================
    static inline void fftshift3d(std::vector<std::complex<double>>& dst,
        const std::vector<std::complex<double>>& src,
        int nx, int ny, int nz) {
        dst.resize(src.size());
        int hx = nx / 2, hy = ny / 2, hz = nz / 2;
        for (int k = 0; k < nz; ++k) {
            int kk = (k + hz) % nz;
            for (int j = 0; j < ny; ++j) {
                int jj = (j + hy) % ny;
                for (int i = 0; i < nx; ++i) {
                    int ii = (i + hx) % nx;
                    dst[idx3(ii, jj, kk, nx, ny)] = src[idx3(i, j, k, nx, ny)];
                }
            }
        }
    }
    static inline void ifftshift3d(std::vector<std::complex<double>>& dst,
        const std::vector<std::complex<double>>& src,
        int nx, int ny, int nz)
    {
        dst.resize(src.size());
        int hx = nx / 2, hy = ny / 2, hz = nz / 2;

        // 逆シフトは「-hx, -hy, -hz」でコピーする
        for (int k = 0; k < nz; ++k) {
            int kk = (k - hz % nz + nz) % nz;  // (k - hz) mod nz
            for (int j = 0; j < ny; ++j) {
                int jj = (j - hy % ny + ny) % ny;
                for (int i = 0; i < nx; ++i) {
                    int ii = (i - hx % nx + nx) % nx;
                    dst[idx3(ii, jj, kk, nx, ny)] = src[idx3(i, j, k, nx, ny)];
                }
            }
        }
    }
    // ========================= 窓関数（コサイン） =========================
    // FFT入力用: 境界でも eps_fft を残す
    static inline double cosine_edge_taper_1d(int dist_to_edge, int t_cells)
    {
        if (dist_to_edge <= 0) return 0.0;
        if (t_cells <= 0) return 1.0;
        if (dist_to_edge < t_cells) {
            return 0.5 * (1.0 - std::cos(M_PI * (double)dist_to_edge / (double)t_cells));
        }
        return 1.0;
    }

    // FFT入力用: 境界でも eps_fft_* を残す（各方向別）
    static inline void make_cosine_window_for_fft(std::vector<double>& W,
        int NX, int NY, int NZ,
        int t_cells_x = 3, int t_cells_y = 3, int t_cells_z = 3,
        double eps_fft_x = 0.05, double eps_fft_y = 0.05, double eps_fft_z = 0.05) {
        W.assign((size_t)NX * NY * NZ, 1.0);
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    const int dx = std::min(i, NX - 1 - i);
                    const int dy = std::min(j, NY - 1 - j);
                    const int dz = std::min(k, NZ - 1 - k);

                    const double wx = eps_fft_x + (1.0 - eps_fft_x) * cosine_edge_taper_1d(dx, t_cells_x);
                    const double wy = eps_fft_y + (1.0 - eps_fft_y) * cosine_edge_taper_1d(dy, t_cells_y);
                    const double wz = eps_fft_z + (1.0 - eps_fft_z) * cosine_edge_taper_1d(dz, t_cells_z);

                    W[idx3(i, j, k, NX, NY)] = wx * wy * wz;
                }
    }

    // 出力摂動用: 境界で必ず 0、内部で 1（各方向別）
    static inline void make_cosine_window_for_output(std::vector<double>& W,
        int NX, int NY, int NZ,
        int t_cells_x = 3, int t_cells_y = 3, int t_cells_z = 3) {
        W.assign((size_t)NX * NY * NZ, 1.0);
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    const int dx = std::min(i, NX - 1 - i);
                    const int dy = std::min(j, NY - 1 - j);
                    const int dz = std::min(k, NZ - 1 - k);

                    const double wx = cosine_edge_taper_1d(dx, t_cells_x);
                    const double wy = cosine_edge_taper_1d(dy, t_cells_y);
                    const double wz = cosine_edge_taper_1d(dz, t_cells_z);

                    W[idx3(i, j, k, NX, NY)] = wx * wy * wz;
                }
    }

    // ========================= k空間乱数場ヘルパ =========================
    static inline std::vector<double> make_gaussian_kernel_1d(int radius_cells)
    {
        radius_cells = std::max(0, radius_cells);
        std::vector<double> kernel((size_t)2 * radius_cells + 1, 1.0);
        if (radius_cells == 0) return kernel;

        const double sigma = std::max(1e-12, radius_cells / 2.0);
        double sum = 0.0;
        for (int t = -radius_cells; t <= radius_cells; ++t) {
            const double w = std::exp(-0.5 * (double(t) * double(t)) / (sigma * sigma));
            kernel[(size_t)(t + radius_cells)] = w;
            sum += w;
        }
        if (sum > 0.0) {
            for (double& v : kernel) v /= sum;
        }
        return kernel;
    }

    static inline void gaussian_smooth3d_inplace(
        std::vector<double>& a,
        int NX, int NY, int NZ,
        int corr_cells_x, int corr_cells_y, int corr_cells_z)
    {
        if (a.empty()) return;

        const int rx = std::max(0, corr_cells_x);
        const int ry = std::max(0, corr_cells_y);
        const int rz = std::max(0, corr_cells_z);

        // If all correlation radii are zero, do not perform any smoothing.
        // This preserves the original independent random field exactly.
        if (rx == 0 && ry == 0 && rz == 0) return;

        auto clamp_index = [](int v, int lo, int hi) {
            return std::max(lo, std::min(hi, v));
            };

        std::vector<double> tmp = a;
        std::vector<double> out = a;

        // X-direction smoothing. If corr_cells_x == 0, skip this direction.
        if (rx > 0) {
            const std::vector<double> kx = make_gaussian_kernel_1d(rx);
            for (int k = 0; k < NZ; ++k)
                for (int j = 0; j < NY; ++j)
                    for (int i = 0; i < NX; ++i) {
                        double s = 0.0;
                        for (int di = -rx; di <= rx; ++di) {
                            const int ii = clamp_index(i + di, 0, NX - 1);
                            s += kx[(size_t)(di + rx)] * a[idx3(ii, j, k, NX, NY)];
                        }
                        tmp[idx3(i, j, k, NX, NY)] = s;
                    }
        }
        else {
            tmp = a;
        }

        // Y-direction smoothing. If corr_cells_y == 0, skip this direction.
        if (ry > 0) {
            const std::vector<double> ky = make_gaussian_kernel_1d(ry);
            for (int k = 0; k < NZ; ++k)
                for (int j = 0; j < NY; ++j)
                    for (int i = 0; i < NX; ++i) {
                        double s = 0.0;
                        for (int dj = -ry; dj <= ry; ++dj) {
                            const int jj = clamp_index(j + dj, 0, NY - 1);
                            s += ky[(size_t)(dj + ry)] * tmp[idx3(i, jj, k, NX, NY)];
                        }
                        out[idx3(i, j, k, NX, NY)] = s;
                    }
        }
        else {
            out = tmp;
        }

        // Z-direction smoothing. If corr_cells_z == 0, skip this direction.
        if (rz > 0) {
            const std::vector<double> kz = make_gaussian_kernel_1d(rz);
            for (int k = 0; k < NZ; ++k)
                for (int j = 0; j < NY; ++j)
                    for (int i = 0; i < NX; ++i) {
                        double s = 0.0;
                        for (int dk = -rz; dk <= rz; ++dk) {
                            const int kk = clamp_index(k + dk, 0, NZ - 1);
                            s += kz[(size_t)(dk + rz)] * out[idx3(i, j, kk, NX, NY)];
                        }
                        a[idx3(i, j, k, NX, NY)] = s;
                    }
        }
        else {
            a = out;
        }
    }

    static inline void normalize_masked_zero_mean_unit_maxabs(
        std::vector<double>& a,
        const std::vector<uint8_t>& mask)
    {
        double mean = 0.0;
        long long cnt = 0;
        for (size_t n = 0; n < a.size(); ++n) {
            if (!mask[n]) continue;
            mean += a[n];
            ++cnt;
        }
        if (cnt > 0) mean /= (double)cnt;

        double maxabs = 0.0;
        for (size_t n = 0; n < a.size(); ++n) {
            if (!mask[n]) {
                a[n] = 0.0;
                continue;
            }
            a[n] -= mean;
            maxabs = std::max(maxabs, std::abs(a[n]));
        }
        if (maxabs > 0.0) {
            for (size_t n = 0; n < a.size(); ++n) {
                if (mask[n]) a[n] /= maxabs;
            }
        }
    }

    // ========================= 低周波微摂動（相関付きロールオフ） =========================
    static inline void draw_correlated_lowfreq(
        std::vector<std::complex<double>>& Fh, // in/out: fftshift 後
        const std::vector<uint8_t>& mask,      // bool相当 (1=更新)
        int NX, int NY, int NZ,
        double epsR, double epst, double attenuation,
        int corr_cells_x, int corr_cells_y, int corr_cells_z,
        std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        const int cx = NX / 2, cy = NY / 2, cz = NZ / 2;

        auto mmax_of = [&](int axis) {
            int mmax = 1;
            for (int k = 0; k < NZ; ++k)
                for (int j = 0; j < NY; ++j)
                    for (int i = 0; i < NX; ++i) {
                        if (!mask[idx3(i, j, k, NX, NY)]) continue;
                        int m = 0;
                        if (axis == 0) m = std::abs(i - cx);
                        if (axis == 1) m = std::abs(j - cy);
                        if (axis == 2) m = std::abs(k - cz);
                        mmax = std::max(mmax, m);
                    }
            return mmax;
            };
        const double mxmax = std::max(1, mmax_of(0));
        const double mymax = std::max(1, mmax_of(1));
        const double mzmax = std::max(1, mmax_of(2));

        std::vector<double> w((size_t)NX * NY * NZ, 0.0);
        const double att = std::clamp(attenuation, 1e-6, 1.0);
        const double gamma = std::sqrt(-std::log(att));
        for (int k = 0; k < NZ; ++k) {
            const double rz = (k - cz) / mzmax;
            for (int j = 0; j < NY; ++j) {
                const double ry = (j - cy) / mymax;
                for (int i = 0; i < NX; ++i) {
                    const double rx = (i - cx) / mxmax;
                    const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                    double ww = std::exp(-(gamma * std::clamp(r, 0.0, 1.0)) * (gamma * std::clamp(r, 0.0, 1.0)));
                    if (r > 1.0) ww = 0.0;
                    w[idx3(i, j, k, NX, NY)] = ww;
                }
            }
        }

        std::vector<double> ramp((size_t)NX * NY * NZ, 0.0);
        std::vector<double> rphase((size_t)NX * NY * NZ, 0.0);
        for (size_t n = 0; n < mask.size(); ++n) {
            if (!mask[n]) continue;
            ramp[n] = U(rng);
            rphase[n] = U(rng);
        }

        gaussian_smooth3d_inplace(ramp, NX, NY, NZ, corr_cells_x, corr_cells_y, corr_cells_z);
        gaussian_smooth3d_inplace(rphase, NX, NY, NZ, corr_cells_x, corr_cells_y, corr_cells_z);
        normalize_masked_zero_mean_unit_maxabs(ramp, mask);
        normalize_masked_zero_mean_unit_maxabs(rphase, mask);

        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    const auto id = idx3(i, j, k, NX, NY);
                    if (!mask[id]) continue;
                    if (i == cx && j == cy && k == cz) continue;
                    const int jx = _sym_idx(i, NX, cx), jy = _sym_idx(j, NY, cy), jz = _sym_idx(k, NZ, cz);
                    if ((i > jx) || (i == jx && j > jy) || (i == jx && j == jy && k > jz)) continue;

                    const double ww = w[id];
                    if (ww <= 0.0) continue;

                    const double a_rand = std::max(1e-6, 1.0 + ww * epsR * ramp[id]);
                    const double th_rand = ww * epst * rphase[id];
                    const std::complex<double> alpha = std::polar(a_rand, th_rand);

                    const auto Fk_old = Fh[id];
                    const auto Fk_new = Fk_old * alpha;
                    if (i == jx && j == jy && k == jz) {
                        Fh[id] = std::complex<double>(Fk_new.real(), 0.0);
                    }
                    else {
                        Fh[id] = Fk_new;
                        Fh[idx3(jx, jy, jz, NX, NY)] = std::conj(Fk_new);
                    }
                }
        Fh[idx3(cx, cy, cz, NX, NY)] = Fh[idx3(cx, cy, cz, NX, NY)];
    }

    // ========================= 低域再フィルタ（x空間→k空間→x空間） =========================
    //static inline void lowpass_again(std::vector<double>& delta, // in/out: 長さ NX*NY*NZ
    //    int NX, int NY, int NZ,
    //    int mxmax, int mymax, int mzmax,
    //    double attenuation = 0.5)
    //{
    //    // FFT
    //    std::vector<std::complex<double>> x(NX * NY * NZ), X, Xs, Xf;
    //    for (size_t n = 0; n < x.size(); ++n) x[n] = std::complex<double>(delta[n], 0.0);
    //    FFT3D fft3(NX, NY, NZ);
    //    fft3.fwd(X, x);
    //    // shift
    //    fftshift3d(Xs, X, NX, NY, NZ);

    //    // 球状ガウス
    //    std::vector<double> w(NX * NY * NZ, 0.0);
    //    int cx = NX / 2, cy = NY / 2, cz = NZ / 2;
    //    double att = std::clamp(attenuation, 1e-6, 1.0);
    //    double gamma = std::sqrt(-std::log(att));
    //    for (int k = 0; k < NZ; ++k) {
    //        double rz = (k - cz) / std::max(1, mzmax);
    //        for (int j = 0; j < NY; ++j) {
    //            double ry = (j - cy) / std::max(1, mymax);
    //            for (int i = 0; i < NX; ++i) {
    //                double rx = (i - cx) / std::max(1, mxmax);
    //                double r = std::sqrt(rx * rx + ry * ry + rz * rz);
    //                double ww = std::exp(-(gamma * std::clamp(r, 0.0, 1.0)) * (gamma * std::clamp(r, 0.0, 1.0)));
    //                if (r > 1.0) ww = 0.0;
    //                w[idx3(i, j, k, NX, NY)] = ww;
    //            }
    //        }
    //    }
    //    Xf.resize(Xs.size());
    //    for (size_t n = 0; n < Xs.size(); ++n) Xf[n] = Xs[n] * w[n];

    //    // ifft
    //    std::vector<std::complex<double>> Xu, xout;
    //    ifftshift3d(Xu, Xf, NX, NY, NZ);
    //    fft3.inv(xout, Xu);
    //    delta.resize(xout.size());
    // //   const double Ntot = double(NX) * NY * NZ;
    //    for (size_t n = 0; n < xout.size(); ++n) delta[n] = xout[n].real();
    //}

    // ========================= 最近傍補間（散布→規則格子） =========================
    struct Pt { double x, y, z, v; };
    struct HashGrid {
        std::array<double, 3> origin{};
        std::array<double, 3> cell{};
        std::array<int, 3> dim{};
        std::vector<int> head, next;
        const std::vector<Pt>* pts = nullptr;

        static int fd(double a, double b) { return (int)std::floor(a / b); }
        static int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }
        inline int lid(int ix, int iy, int iz)const { return (iz * dim[1] + iy) * dim[0] + ix; }

        void build(const std::vector<Pt>& P, std::array<double, 3> cellSize, int pad = 2) {
            pts = &P; cell = cellSize;
            std::array<double, 3> mn{ +INFINITY,+INFINITY,+INFINITY }, mx{ -INFINITY,-INFINITY,-INFINITY };
            for (auto& p : P) {
                mn[0] = std::min(mn[0], p.x); mn[1] = std::min(mn[1], p.y); mn[2] = std::min(mn[2], p.z);
                mx[0] = std::max(mx[0], p.x); mx[1] = std::max(mx[1], p.y); mx[2] = std::max(mx[2], p.z);
            }
            for (int a = 0; a < 3; ++a) {
                origin[a] = mn[a] - pad * cell[a];
                double ext = (mx[a] - mn[a]) + 2 * pad * cell[a];
                dim[a] = std::max(1, (int)std::ceil(ext / cell[a]));
            }
            head.assign(dim[0] * dim[1] * dim[2], -1);
            next.assign(P.size(), -1);
            for (int i = 0; i < (int)P.size(); ++i) {
                int ix = clampi(fd(P[i].x - origin[0], cell[0]), 0, dim[0] - 1);
                int iy = clampi(fd(P[i].y - origin[1], cell[1]), 0, dim[1] - 1);
                int iz = clampi(fd(P[i].z - origin[2], cell[2]), 0, dim[2] - 1);
                int id = lid(ix, iy, iz);
                next[i] = head[id]; head[id] = i;
            }
        }

        int nearest(double x, double y, double z, double r0, double rcap)const {
            if (!pts || pts->empty()) return -1;
            double best = std::numeric_limits<double>::infinity(); int bestId = -1;
            double r = std::max(r0, 1e-12);
            while (r <= rcap) {
                int ix0 = clampi(fd((x - origin[0] - r), cell[0]), 0, dim[0] - 1);
                int iy0 = clampi(fd((y - origin[1] - r), cell[1]), 0, dim[1] - 1);
                int iz0 = clampi(fd((z - origin[2] - r), cell[2]), 0, dim[2] - 1);
                int ix1 = clampi(fd((x - origin[0] + r), cell[0]), 0, dim[0] - 1);
                int iy1 = clampi(fd((y - origin[1] + r), cell[1]), 0, dim[1] - 1);
                int iz1 = clampi(fd((z - origin[2] + r), cell[2]), 0, dim[2] - 1);
                double r2 = r * r;
                for (int iz = iz0; iz <= iz1; ++iz)
                    for (int iy = iy0; iy <= iy1; ++iy)
                        for (int ix = ix0; ix <= ix1; ++ix) {
                            int id = lid(ix, iy, iz);
                            for (int p = head[id]; p != -1; p = next[p]) {
                                const auto& q = (*pts)[p];
                                double dx = q.x - x, dy = q.y - y, dz = q.z - z;
                                double d2 = dx * dx + dy * dy + dz * dz;
                                if (d2 <= r2 && d2 < best) { best = d2; bestId = p; }
                            }
                        }
                if (bestId != -1) break;
                r *= 2.0;
            }
            if (bestId == -1) { // 全探索保険
                for (int i = 0; i < (int)pts->size(); ++i) {
                    const auto& q = (*pts)[i];
                    double dx = q.x - x, dy = q.y - y, dz = q.z - z;
                    double d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < best) { best = d2; bestId = i; }
                }
            }
            return bestId;
        }
    };
    static inline void sample_idw_to_grid(
        const std::vector<Pt>& pts, const GridSpec& g, std::vector<double>& out_values)
    {
        out_values.assign((size_t)g.nx * g.ny * g.nz, std::numeric_limits<double>::quiet_NaN());
        if (pts.empty()) return;

        constexpr int    K = 12;
        constexpr double P = 2.0;
        constexpr double EPS = 1e-12;
        const double halfP = 0.5 * P;

        // クエリ格子点の座標（セル中心を採用）
        auto qcoord = [&](int i, int j, int k) -> std::array<double, 3> {
            return { g.ox + (i + 0.5) * g.dx,
                     g.oy + (j + 0.5) * g.dy,
                     g.oz + (k + 0.5) * g.dz };
            };

        // 近傍保持用（距離^2 と インデックス）
        struct Node { double d2; size_t idx; };

        // K近傍収集（外部依存なし、ナイーブ探索）
        auto gather_knn = [&](const std::array<double, 3>& q, std::vector<Node>& knn) {
            knn.clear();
            knn.reserve((size_t)std::min<int>(K, (int)pts.size()));

            auto push_heap_max = [](std::vector<Node>& a) {
                std::push_heap(a.begin(), a.end(),
                    [](const Node& A, const Node& B) { return A.d2 < B.d2; }); // max-heap
                };
            auto pop_heap_max = [](std::vector<Node>& a) {
                std::pop_heap(a.begin(), a.end(),
                    [](const Node& A, const Node& B) { return A.d2 < B.d2; });
                };

            // 距離は各軸を格子間隔で正規化（等方距離だとzのスケール差で歪むため）
            const double invdx = (g.dx > 0) ? 1.0 / g.dx : 1.0;
            const double invdy = (g.dy > 0) ? 1.0 / g.dy : 1.0;
            const double invdz = (g.dz > 0) ? 1.0 / g.dz : 1.0;

            for (size_t t = 0; t < pts.size(); ++t) {
                const double dx = (pts[t].x - q[0]) * invdx;
                const double dy = (pts[t].y - q[1]) * invdy;
                const double dz = (pts[t].z - q[2]) * invdz;
                const double d2 = dx * dx + dy * dy + dz * dz;

                if ((int)knn.size() < K) {
                    knn.push_back({ d2, t });
                    if ((int)knn.size() == K) push_heap_max(knn);
                }
                else if (d2 < knn.front().d2) {
                    pop_heap_max(knn);
                    knn.back() = { d2, t };
                    push_heap_max(knn);
                }
            }
            std::sort(knn.begin(), knn.end(), [](const Node& a, const Node& b) { return a.d2 < b.d2; });
            };

        std::vector<Node> knn;
        knn.reserve((size_t)std::min<int>(K, (int)pts.size()));

        const double eps2 = EPS * EPS;

        for (int k = 0; k < g.nz; ++k) {
            for (int j = 0; j < g.ny; ++j) {
                for (int i = 0; i < g.nx; ++i) {
                    const auto q = qcoord(i, j, k);

                    // 完全一致（距離ゼロ）なら即値を採用（安定化）
                    bool same = false;
                    for (const auto& p : pts) {
                        double dx = (p.x - q[0]) / (g.dx > 0 ? g.dx : 1.0);
                        double dy = (p.y - q[1]) / (g.dy > 0 ? g.dy : 1.0);
                        double dz = (p.z - q[2]) / (g.dz > 0 ? g.dz : 1.0);
                        if (dx * dx + dy * dy + dz * dz < eps2) {
                            out_values[idx3(i, j, k, g.nx, g.ny)] = p.v;
                            same = true;
                            break;
                        }
                    }
                    if (same) continue;

                    // K近傍を集めて IDW 合成
                    gather_knn(q, knn);
                    if (knn.empty()) continue;

                    double wsum = 0.0, vsum = 0.0;
                    for (const auto& n : knn) {
                        const double w = (n.d2 < eps2) ? 1.0 / EPS : std::pow(n.d2, -halfP);
                        wsum += w;
                        vsum += w * pts[n.idx].v;
                    }
                    out_values[idx3(i, j, k, g.nx, g.ny)] = (wsum > 0.0) ? (vsum / wsum)
                        : std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }
    static inline void sample_nearest_to_grid(
        const std::vector<Pt>& pts, const GridSpec& g, std::vector<double>& out_values)
    {
        out_values.resize((size_t)g.nx * g.ny * g.nz);
        HashGrid hg; hg.build(pts, { g.dx,g.dy,g.dz }, 2);
        double r0 = 1.5 * std::min({ g.dx,g.dy,g.dz });
        double rcap = r0 * 64;

        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    double x = g.ox + i * g.dx, y = g.oy + j * g.dy, z = g.oz + k * g.dz;
                    int id = hg.nearest(x, y, z, r0, rcap);
                    out_values[idx3(i, j, k, g.nx, g.ny)] = (id >= 0 ? pts[id].v : std::numeric_limits<double>::quiet_NaN());
                }
    }

    // ========================= トリリニア補間（RegularGridInterpolator 相当） =========================
    static inline double trilinear_sample(const std::vector<double>& grid,
        const GridSpec& g, double x, double y, double z)
    {
        // sample_idw_to_grid() / sample_nearest_to_grid() では格子値を
        // 「セル中心」(ox + (i+0.5)*dx) 上に配置している。
        // そのため、補間側も同じセル中心基準にそろえないと半セルずれが入る。
        double fx = (x - (g.ox + 0.5 * g.dx)) / g.dx;
        double fy = (y - (g.oy + 0.5 * g.dy)) / g.dy;
        double fz = (z - (g.oz + 0.5 * g.dz)) / g.dz;
        int i = (int)std::floor(fx), j = (int)std::floor(fy), k = (int)std::floor(fz);
        if (i < 0 || i + 1 >= g.nx || j < 0 || j + 1 >= g.ny || k < 0 || k + 1 >= g.nz) return std::numeric_limits<double>::quiet_NaN();
        double tx = fx - i, ty = fy - j, tz = fz - k;
        auto id = [&](int ii, int jj, int kk) { return idx3(ii, jj, kk, g.nx, g.ny); };
        double c000 = grid[id(i, j, k)];
        double c100 = grid[id(i + 1, j, k)];
        double c010 = grid[id(i, j + 1, k)];
        double c110 = grid[id(i + 1, j + 1, k)];
        double c001 = grid[id(i, j, k + 1)];
        double c101 = grid[id(i + 1, j, k + 1)];
        double c011 = grid[id(i, j + 1, k + 1)];
        double c111 = grid[id(i + 1, j + 1, k + 1)];
        double c00 = c000 * (1 - tx) + c100 * tx, c10 = c010 * (1 - tx) + c110 * tx;
        double c01 = c001 * (1 - tx) + c101 * tx, c11 = c011 * (1 - tx) + c111 * tx;
        double c0 = c00 * (1 - ty) + c10 * ty, c1 = c01 * (1 - ty) + c11 * ty;
        return c0 * (1 - tz) + c1 * tz;
    }

    // ========================= QR 直交化（列） =========================
    static inline std::vector<double> orth_to_cols_qr(
        const std::vector<double>& v, const std::vector<std::vector<double>>& cols, double eps = 1e-12, bool normalize = true)
    {
        if (cols.empty()) {
            std::vector<double> r = v; if (normalize) normalize_inplace(r); return r;
        }
        int n = (int)v.size(), m = 0;
        for (auto& c : cols) if ((int)c.size() == n) ++m;
        Eigen::MatrixXd C(n, m); int jj = 0;
        for (auto& c : cols) {
            if ((int)c.size() != n) continue;
            double nrm = std::sqrt(std::inner_product(c.begin(), c.end(), c.begin(), 0.0));
            if (nrm <= eps) continue;
            for (int i = 0; i < n; ++i) C(i, jj) = c[i];
            ++jj;
        }
        if (jj == 0) { std::vector<double> r = v; if (normalize) normalize_inplace(r); return r; }
        C.conservativeResize(n, jj);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(C);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, jj);
        Eigen::VectorXd ev(n);
        for (int i = 0; i < n; ++i) ev(i) = v[i];
        Eigen::VectorXd proj = Q * (Q.transpose() * ev);
        Eigen::VectorXd vv = ev - proj;
        std::vector<double> out(n);
        double nrm = vv.norm();
        if (normalize && nrm > eps) vv /= nrm;
        for (int i = 0; i < n; ++i) out[i] = vv(i);
        return out;
    }


    static inline double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) throw std::runtime_error("dot_product size mismatch");
        long double s = 0.0L;
        for (size_t i = 0; i < a.size(); ++i) s += (long double)a[i] * (long double)b[i];
        return (double)s;
    }

    static inline double cosine_similarity_abs(const std::vector<double>& a, const std::vector<double>& b, double eps = 1.0e-20) {
        if (a.size() != b.size()) throw std::runtime_error("cosine_similarity_abs size mismatch");
        const double na = l2norm(a);
        const double nb = l2norm(b);
        if (na <= eps || nb <= eps) return 0.0;
        return std::abs(dot_product(a, b)) / (na * nb);
    }

    static inline void axpy_inplace(std::vector<double>& y, double alpha, const std::vector<double>& x) {
        if (y.size() != x.size()) throw std::runtime_error("axpy_inplace size mismatch");
        for (size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
    }

    static inline std::vector<double> linear_combination_from_columns(
        const std::vector<std::vector<double>>& cols,
        const Eigen::VectorXd& coeffs)
    {
        if ((int)cols.size() != coeffs.size()) {
            throw std::runtime_error("linear_combination_from_columns size mismatch");
        }
        if (cols.empty()) return {};
        std::vector<double> out(cols.front().size(), 0.0);
        for (int j = 0; j < coeffs.size(); ++j) {
            axpy_inplace(out, coeffs(j), cols[(size_t)j]);
        }
        return out;
    }

    static inline Eigen::MatrixXd vectors_to_column_matrix(const std::vector<std::vector<double>>& cols) {
        if (cols.empty()) return Eigen::MatrixXd(0, 0);
        const int n = (int)cols.front().size();
        const int m = (int)cols.size();
        Eigen::MatrixXd M(n, m);
        for (int j = 0; j < m; ++j) {
            if ((int)cols[(size_t)j].size() != n) throw std::runtime_error("vectors_to_column_matrix size mismatch");
            for (int i = 0; i < n; ++i) M(i, j) = cols[(size_t)j][(size_t)i];
        }
        return M;
    }

    static inline Eigen::MatrixXd vectors_to_row_matrix(const std::vector<std::vector<double>>& rows) {
        if (rows.empty()) return Eigen::MatrixXd(0, 0);
        const int ncol = (int)rows.front().size();
        const int nrow = (int)rows.size();
        Eigen::MatrixXd M(nrow, ncol);
        for (int i = 0; i < nrow; ++i) {
            if ((int)rows[(size_t)i].size() != ncol) throw std::runtime_error("vectors_to_row_matrix size mismatch");
            for (int j = 0; j < ncol; ++j) M(i, j) = rows[(size_t)i][(size_t)j];
        }
        return M;
    }

    static inline std::pair<double, double> minmax_value(const std::vector<double>& v) {
        if (v.empty()) return { 0.0, 0.0 };
        auto mm = std::minmax_element(v.begin(), v.end());
        return { *mm.first, *mm.second };
    }

    static inline void ensure_finite_vector_or_throw(
        const std::vector<double>& v,
        const std::string& name,
        int col = -1)
    {
        for (size_t i = 0; i < v.size(); ++i) {
            if (!std::isfinite(v[i])) {
                std::ostringstream oss;
                oss << "Non-finite value detected in " << name
                    << " at index=" << i;
                if (col >= 0) oss << ", column=" << col;
                oss << ", value=" << v[i];
                throw std::runtime_error(oss.str());
            }
        }
    }

    static inline void ensure_finite_matrix_or_throw(
        const Eigen::MatrixXd& M,
        const std::string& name)
    {
        for (int r = 0; r < M.rows(); ++r) {
            for (int c = 0; c < M.cols(); ++c) {
                const double v = M(r, c);
                if (!std::isfinite(v)) {
                    std::ostringstream oss;
                    oss << "Non-finite value detected in " << name
                        << " at row=" << r << ", col=" << c
                        << ", value=" << v;
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }

    static inline void ensure_finite_eigen_vector_or_throw(
        const Eigen::VectorXd& v,
        const std::string& name)
    {
        for (int i = 0; i < v.size(); ++i) {
            if (!std::isfinite(v(i))) {
                std::ostringstream oss;
                oss << "Non-finite value detected in " << name
                    << " at index=" << i
                    << ", value=" << v(i);
                throw std::runtime_error(oss.str());
            }
        }
    }


    static inline std::vector<std::vector<double>> column_matrix_to_vectors(const Eigen::MatrixXd& M) {
        std::vector<std::vector<double>> cols;
        if (M.rows() == 0 || M.cols() == 0) return cols;
        cols.assign((size_t)M.cols(), std::vector<double>((size_t)M.rows(), 0.0));
        for (int j = 0; j < M.cols(); ++j) {
            for (int i = 0; i < M.rows(); ++i) cols[(size_t)j][(size_t)i] = M(i, j);
        }
        return cols;
    }


    // ========================= 割線ラインサーチ（両側） =========================
    // ここでは _eval_RMS_at_alpha_no_bounds を外部関数ポインタで渡す
//    using EvalRMS = std::function<double(double)>;
//
//    static inline double secant_one_side(double sign, EvalRMS eval_RMS,
//        double RMS0, double lower, double upper, double step0 = 1.0,
//        int max_iter = 20, double tol_rel = 1e-1, double eps_den = 1e-14, double max_alpha = 1e6)
//    {
//        static double a_pre = 0.0;
//        auto mid = 0.5 * (lower + upper);
//        auto evalS = [&](double a) { return eval_RMS(sign * a); };
//
//        double a0 = 0.0, g0 = RMS0;
//        double a1 = std::max(a0 + std::max(a_pre, std::max(1e-12, std::abs(step0))), 1e-12);
//        double g1 = evalS(a1);
//
//        for (int it = 0; it < max_iter; ++it) {
//
//            if (lower <= g1 && g1 <= upper) return sign * a1;
//            double den = (g1 - g0);
//            double a2;
//            if (std::abs(den) < eps_den) a2 = 0.5 * (a1 + a0);
//            else a2 = a1 - (g1 - mid) * (a1 - a0) / den;
//            if (!std::isfinite(a2)) a2 = 0.5 * (a1 + a0);
//            a2 = std::clamp(a2, 0.0, max_alpha);
//            if (a2 <= a0 || std::abs(a2 - a1) < 1e-15) a2 = 0.5 * (a1 + a0);
//            a0 = a1; g0 = g1;
//            a1 = a2; g1 = evalS(a1);
//            cout << "In Line Search, Iteration is:" << it << endl;
//            cout << "In Line Search, a1 is:" << a1 << endl;
//            cout << "In Line Search, RMS, RMS_lower, RMS_upper are:" << g1<<" "<< lower<<" "<< upper << endl;
//        }
//        a_pre = a1;
//        return sign * a1;
//    }
//
    using EvalRMS = std::function<double(double)>;
    static inline double secant_one_side(double sign, EvalRMS eval_RMS,
        double RMS0, double lower, double upper, double step0 = 1.0,
        int max_iter = 20, double tol_rel = 1e-1, double eps_den = 1e-14, double max_alpha = 1e6)
    {
        using std::abs;
        using std::clamp;
        using std::isfinite;
        using std::max;
        using std::min;

        static double a_pre = 0.0;               // 前回の結果を初期拡張に利用
        const double mid = 0.5 * (lower + upper);
        auto evalS = [&](double a) { return eval_RMS(sign * a); };

        // a=0 で既にレンジ内なら即返す
        if (lower <= RMS0 && RMS0 <= upper) {
            return 0.0;
        }

        // 目的関数：g(a)=RMS(sign*a)-mid
        auto g = [&](double a) -> double { return evalS(a) - mid; };

        // 端点チェック
        double aL = 0.0;
        double fL = RMS0 - mid;
        if (!isfinite(fL)) fL = 0.0;
        if (fL == 0.0) return 0.0;

        // 右方向に指数拡張して符号反転ブラケット [aL, aR] を作る
        double aR = clamp(max({ 1e-12, abs(step0), a_pre }), 1e-12, max_alpha);
        double fR = g(aR);

        // 途中でレンジ内に入れば即返す（元の挙動を踏襲）
        if (lower <= (fR + mid) && (fR + mid) <= upper) {
            std::cout << "In Line Search, Iteration is: " << 0 << std::endl;
            std::cout << "In Line Search, a1 is: " << aR << std::endl;
            std::cout << "In Line Search, RMS, RMS_lower, RMS_upper are: "
                << (fR + mid) << " " << lower << " " << upper << std::endl;
            a_pre = aR;
            return sign * aR;
        }

        // 符号反転がなければ右へ拡張
        const int max_expand = max_iter * 3;
        for (int k = 0; k < max_expand && fL * fR > 0.0; ++k) {
            double step = max(1e-12, aR * 0.6);
            aR = min(max_alpha, aR + max(1.0, 1.6 * (k + 1)) * step);
            if (aR <= aL + 1e-15) aR = min(max_alpha, aL + 1e-6);
            fR = g(aR);

            std::cout << "In Line Search (expand), a1=" << aR
                << ", RMS=" << (fR + mid)
                << ", range=[" << lower << ", " << upper << "]" << std::endl;

            if (lower <= (fR + mid) && (fR + mid) <= upper) {
                a_pre = aR;
                return sign * aR;
            }
            if (aR >= max_alpha - 1e-12) break;
        }

        // まだ符号反転が無い場合、aR を縮めて試す（最後のあがき）
        if (fL * fR > 0.0) {
            double aR_try = aR;
            for (int k = 0; k < max_expand && fL * fR > 0.0; ++k) {
                aR_try *= 0.5;
                if (aR_try < 1e-12) break;
                fR = g(aR_try);
                if (lower <= (fR + mid) && (fR + mid) <= upper) {
                    a_pre = aR_try;
                    return sign * aR_try;
                }
            }
            // ブラケット作れなければ現状ベスト（右端）を返す（従来と同程度のフォールバック）
            a_pre = aR;
            return sign * a_pre;
        }

        // ここまでで g(aL)*g(aR) <= 0 が保証
        // 純粋な二分法で g(a)=0 を解く（途中でレンジ内に入れば即返す）
        for (int it = 0; it < max_iter; ++it) {
            double aM = 0.5 * (aL + aR);
            double fM = g(aM);
            double RMSM = fM + mid;

            std::cout << "In Line Search, Iteration is: " << it << std::endl;
            std::cout << "In Line Search, a1 is: " << aM << std::endl;
            std::cout << "In Line Search, RMS, RMS_lower, RMS_upper are: "
                << RMSM << " " << lower << " " << upper << std::endl;

            // レンジ内なら即返す
            if (lower <= RMSM && RMSM <= upper) {
                a_pre = aM;
                return sign * aM;
            }

            // 収束判定（a の幅）
            double width = aR - aL;
            double tol_a = max(1e-12, tol_rel * width);
            if (width <= tol_a || fM == 0.0) {
                a_pre = aM;
                return sign * aM;
            }

            // 次の区間を選ぶ
            if (fL * fM < 0.0) {
                aR = aM; fR = fM;
            }
            else {
                aL = aM; fL = fM;
            }
        }

        // 反復上限：0を返す
        //double aM = 0.5 * (aL + aR);
        //a_pre = aM;
        const double aM = 0.5 * (aL + aR);
        a_pre = aM;
        return sign * aM;
    }

    struct HexCell { std::array<int, 8> conn; };
    static inline void write_legacy_ugrid_vtk(
        const std::string& out_path,
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<HexCell>& cells,
        const std::map<std::string, std::vector<double>>& cell_arrays)
    {
        std::ofstream f(out_path);
        if (!f) throw std::runtime_error("cannot open vtk file");
        f << "# vtk DataFile Version 3.0\n";
        f << "unstructured grid\nASCII\n";
        f << "DATASET UNSTRUCTURED_GRID\n";
        f << "POINTS " << points.size() << " float\n";
        for (auto& p : points) f << (float)p.x() << " " << (float)p.y() << " " << (float)p.z() << "\n";
        // Cells
        int nCells = (int)cells.size();
        int listSize = nCells * (1 + 8);
        f << "CELLS " << nCells << " " << listSize << "\n";
        for (auto& c : cells) {
            f << 8;
            for (int i = 0; i < 8; ++i) f << " " << c.conn[i];
            f << "\n";
        }
        f << "CELL_TYPES " << nCells << "\n";
        for (int i = 0; i < nCells; ++i) f << 12 << "\n"; // VTK_HEXAHEDRON
        // Cell data
        f << "CELL_DATA " << nCells << "\n";
        for (auto& kv : cell_arrays) {
            const auto& name = kv.first; const auto& arr = kv.second;
            if ((int)arr.size() != nCells) continue;
            f << "SCALARS " << name << " float 1\n";
            f << "LOOKUP_TABLE default\n";
            for (double v : arr) f << (float)v << "\n";
        }
    }

    static inline std::string sanitize_vtk_name(const std::string& s)
    {
        std::string out = s;
        for (char& c : out) {
            const bool ok =
                (c >= '0' && c <= '9') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= 'a' && c <= 'z') ||
                (c == '_');
            if (!ok) c = '_';
        }
        if (out.empty()) out = "array";
        return out;
    }

    template <class ElemPtrVec>
    static void write_direction_vtk_from_elements(
        const std::string& out_path,
        const ElemPtrVec& elements,
        const std::vector<std::vector<double>>& D_cols,
        const std::vector<double>* rho_log10 = nullptr)
    {
        if (elements.empty()) {
            throw std::runtime_error("write_direction_vtk_from_elements: elements is empty.");
        }
        if (D_cols.empty()) {
            throw std::runtime_error("write_direction_vtk_from_elements: D_cols is empty.");
        }

        const int nCells = (int)elements.size();

        for (size_t j = 0; j < D_cols.size(); ++j) {
            if ((int)D_cols[j].size() != nCells) {
                throw std::runtime_error("write_direction_vtk_from_elements: D_cols size mismatch.");
            }
        }
        if (rho_log10 && (int)rho_log10->size() != nCells) {
            throw std::runtime_error("write_direction_vtk_from_elements: rho_log10 size mismatch.");
        }

        std::vector<Eigen::Vector3d> points;
        std::vector<HexCell> cells;
        std::map<std::string, std::vector<double>> cell_arrays;

        points.reserve((size_t)nCells * 8);
        cells.reserve((size_t)nCells);

        for (size_t j = 0; j < D_cols.size(); ++j) {
            cell_arrays[sanitize_vtk_name("d_" + std::to_string(j))] = D_cols[j];
        }
        if (rho_log10) {
            cell_arrays["log10rho_base"] = *rho_log10;
        }

        for (int icell = 0; icell < nCells; ++icell) {
            const auto* el = elements[(size_t)icell];
            HexCell hc;

            for (int m = 0; m < 8; ++m) {
                hc.conn[m] = (int)points.size();
                points.push_back(el->nodes[m]->x);
            }
            cells.push_back(hc);
        }

        write_legacy_ugrid_vtk(out_path, points, cells, cell_arrays);
    }
}

void Analysis::Analysis::RunFFTSensitivityAnalysis() {

    // NOTE:
    // Separate windows are used for
    //   1) FFT input       : W_fft (boundary keeps eps_window_fft)
    //   2) output delta    : W_out (boundary goes exactly to 0)
    // cells_window_out_num should be provided from settings / header.
    // If not set (>0), it falls back to cells_window.

    // --------- 1) 層ごとの平均高さ → 最上層を z=0 にシフト ----------
    std::vector<double> averageZs(sameIDZElements.size(), 0.0);
    int minIDZ_invElems = std::numeric_limits<int>::max();

    for (size_t i = 0; i < sameIDZElements.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < sameIDZElements[i].size(); ++j) {
            const auto* e = sameIDZElements[i][j];
            sum += e->centerCoord.coeff(2);
            if (e->isInvertedElement)
                minIDZ_invElems = std::min(minIDZ_invElems, e->IDZ);
        }
        const double denom = std::max<size_t>(1, sameIDZElements[i].size());
        averageZs[i] = sum / denom;
    }
    if (minIDZ_invElems == std::numeric_limits<int>::max())
        minIDZ_invElems = 0;
    minIDZ_invElems = std::clamp(minIDZ_invElems, 0, (int)averageZs.size() - 1);

    const double zeroLevelHeight = averageZs[minIDZ_invElems];
    for (double& z : averageZs) z -= zeroLevelHeight;
    for (double& z : averageZs) cout << z << endl;

    // --------- 2) 散布点（log10ρ） ----------
    // NOTE:
    // FFT sampling uses physical x/y/z coordinates in this version.
    // The generated perturbation is smooth on a uniform physical-coordinate FFT grid
    // and is interpolated back to the original inverted elements using physical coordinates.
    constexpr bool use_ijk_xyz_for_fft = false;

    auto make_sorted_unique_coords = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        std::vector<double> u;
        u.reserve(v.size());
        const double abs_tol = 1.0e-8;
        const double rel_tol = 1.0e-10;
        for (double x : v) {
            if (u.empty()) {
                u.push_back(x);
                continue;
            }
            const double tol = abs_tol + rel_tol * std::max(std::abs(x), std::abs(u.back()));
            if (std::abs(x - u.back()) > tol) {
                u.push_back(x);
            }
        }
        return u;
        };

    auto nearest_sorted_index = [](const std::vector<double>& u, double x) -> int {
        if (u.empty()) return 0;
        auto it = std::lower_bound(u.begin(), u.end(), x);
        if (it == u.begin()) return 0;
        if (it == u.end()) return (int)u.size() - 1;
        const int hi = (int)std::distance(u.begin(), it);
        const int lo = hi - 1;
        return (std::abs(u[(size_t)hi] - x) < std::abs(x - u[(size_t)lo])) ? hi : lo;
        };

    std::vector<double> x_centers_all;
    std::vector<double> y_centers_all;
    x_centers_all.reserve(invertedRhoIDToElementVector.size());
    y_centers_all.reserve(invertedRhoIDToElementVector.size());

    int minIDZ_sample = std::numeric_limits<int>::max();
    int maxIDZ_sample = std::numeric_limits<int>::min();
    for (size_t i = 0; i < invertedRhoIDToElementVector.size(); ++i) {
        const auto* el = invertedRhoIDToElementVector[i];
        x_centers_all.push_back(el->centerCoord.coeff(0));
        y_centers_all.push_back(el->centerCoord.coeff(1));
        const int idz = std::clamp(el->IDZ, 0, (int)averageZs.size() - 1);
        minIDZ_sample = std::min(minIDZ_sample, idz);
        maxIDZ_sample = std::max(maxIDZ_sample, idz);
    }
    if (minIDZ_sample == std::numeric_limits<int>::max()) {
        minIDZ_sample = minIDZ_invElems;
        maxIDZ_sample = minIDZ_invElems;
    }

    const std::vector<double> x_unique = make_sorted_unique_coords(x_centers_all);
    const std::vector<double> y_unique = make_sorted_unique_coords(y_centers_all);
    const int n_ix_sample = std::max(1, (int)x_unique.size());
    const int n_iy_sample = std::max(1, (int)y_unique.size());
    const int n_iz_sample = std::max(1, maxIDZ_sample - minIDZ_sample + 1);

    std::vector<Pt> samples;
    samples.reserve(invertedRhoIDToElementVector.size());
    for (size_t i = 0; i < invertedRhoIDToElementVector.size(); ++i) {
        const auto* el = invertedRhoIDToElementVector[i];
        const int idz = std::clamp(el->IDZ, 0, (int)averageZs.size() - 1);

        const double x = use_ijk_xyz_for_fft
            ? static_cast<double>(nearest_sorted_index(x_unique, el->centerCoord.coeff(0)))
            : el->centerCoord.coeff(0);
        const double y = use_ijk_xyz_for_fft
            ? static_cast<double>(nearest_sorted_index(y_unique, el->centerCoord.coeff(1)))
            : el->centerCoord.coeff(1);
        const double z = use_ijk_xyz_for_fft
            ? static_cast<double>(idz - minIDZ_sample)
            : averageZs[idz];
        const double v = std::log10(el->resistivity);
        samples.push_back({ x,y,z,v });
    }

    // --------- 3) 規則格子（FFT用） ----------
    // Physical-coordinate FFT grid. Nx/Ny/Nz mean the number of regular sampling
    // points between minX/maxX, minY/maxY, and minZ/maxZ.
    const double x_fft_extent = use_ijk_xyz_for_fft
        ? static_cast<double>(n_ix_sample)
        : (maxX - minX);
    const double y_fft_extent = use_ijk_xyz_for_fft
        ? static_cast<double>(n_iy_sample)
        : (maxY - minY);
    const double z_fft_extent = use_ijk_xyz_for_fft
        ? static_cast<double>(n_iz_sample)
        : (maxZ - minZ);

    const double x_fft_dx = x_fft_extent / double(Nx);
    const double y_fft_dy = y_fft_extent / double(Ny);
    const double z_fft_dz = z_fft_extent / double(Nz);

    const double x_fft_ox = use_ijk_xyz_for_fft ? (-0.5 * x_fft_dx) : minX;
    const double y_fft_oy = use_ijk_xyz_for_fft ? (-0.5 * y_fft_dy) : minY;
    const double z_fft_oz = use_ijk_xyz_for_fft ? (-0.5 * z_fft_dz) : minZ;

    GridSpec G{
        Nx, Ny, Nz,
        x_fft_ox, y_fft_oy, z_fft_oz,
        x_fft_dx, y_fft_dy, z_fft_dz
    };

    std::cout << "FFT sampling coordinates: "
        << (use_ijk_xyz_for_fft ? "x/y/z=IJK index space" : "x/y/z=physical space")
        << ", origin=(" << G.ox << ", " << G.oy << ", " << G.oz << ")"
        << ", spacing=(" << G.dx << ", " << G.dy << ", " << G.dz << ")"
        << ", sampled index counts=(" << n_ix_sample << ", " << n_iy_sample << ", " << n_iz_sample << ")"
        << ", sampled IDZ range=[" << minIDZ_sample << ", " << maxIDZ_sample << "]"
        << std::endl;

    std::vector<double> F;
    sample_idw_to_grid(samples, G, F);

    // NaN→平均で埋め & 平均差し引き（DC抑制）
    double mean = 0.0; int cnt = 0;
    for (double v : F) { if (std::isfinite(v)) { mean += v; ++cnt; } }
    mean = (cnt > 0 ? mean / cnt : 0.0);
    for (double& v : F) { if (!std::isfinite(v)) v = mean; v -= mean; }

    // --------- 4) FFT & マスク & 窓（固定前処理） ----------
    std::vector<double> W_fft, W_out;

    const int cells_window_fft_x = (this->cells_window_fft_x >= 0) ? this->cells_window_fft_x : cells_window;
    const int cells_window_fft_y = (this->cells_window_fft_y >= 0) ? this->cells_window_fft_y : cells_window;
    const int cells_window_fft_z = (this->cells_window_fft_z >= 0) ? this->cells_window_fft_z : cells_window;

    const int cells_window_out_x = (this->cells_window_out_x >= 0) ? this->cells_window_out_x : (3 * cells_window_fft_x);
    const int cells_window_out_y = (this->cells_window_out_y >= 0) ? this->cells_window_out_y : (3 * cells_window_fft_y);
    const int cells_window_out_z = (this->cells_window_out_z >= 0) ? this->cells_window_out_z : (3 * cells_window_fft_z);

    const double eps_window_fft_x = (this->eps_window_fft_x >= 0.0) ? this->eps_window_fft_x : eps_window;
    const double eps_window_fft_y = (this->eps_window_fft_y >= 0.0) ? this->eps_window_fft_y : eps_window;
    const double eps_window_fft_z = (this->eps_window_fft_z >= 0.0) ? this->eps_window_fft_z : eps_window;

    make_cosine_window_for_fft(
        W_fft, G.nx, G.ny, G.nz,
        cells_window_fft_x, cells_window_fft_y, cells_window_fft_z,
        eps_window_fft_x, eps_window_fft_y, eps_window_fft_z);
    make_cosine_window_for_output(
        W_out, G.nx, G.ny, G.nz,
        cells_window_out_x, cells_window_out_y, cells_window_out_z);

    FFT3D fft3(G.nx, G.ny, G.nz);
    std::vector<std::complex<double>> x(G.nx * G.ny * G.nz), X, Xs_base;
    for (size_t n = 0; n < F.size(); ++n) x[n] = std::complex<double>(F[n], 0.0) * W_fft[n];

    double norm = 0.0;
    for (int i = 0; i < (int)x.size(); i++) {
        norm += (x[(size_t)i] * std::conj(x[(size_t)i])).real();
    }
    cout << "before fft3_X:" << norm << endl;
    fft3.fwd(X, x);
    fftshift3d(Xs_base, X, G.nx, G.ny, G.nz);
    norm = 0.0;
    for (int i = 0; i < (int)Xs_base.size(); i++) {
        norm += (Xs_base[(size_t)i] * std::conj(Xs_base[(size_t)i])).real();
    }
    cout << "after fftshift3d:" << norm << endl;

    // Low-wavenumber mask.
    // K was previously used as a spherical diameter-like parameter.
    // Here we allow independent cutoffs for x/y/z.  By default they are
    // set to K, so the result is equivalent to the old isotropic setting.
    // If you want to expose these to the input file later, replace these
    // three local values with this->Kx_mask, this->Ky_mask, this->Kz_mask
    // after adding them to Analysis.h / ReadData.
    const int Kx_mask = Kx;
    const int Ky_mask = Ky;
    const int Kz_mask = Kz;

    const int kx_half = std::max(1, Kx_mask);
    const int ky_half = std::max(1, Ky_mask);
    const int kz_half = std::max(1, Kz_mask);

    const int cx = G.nx / 2, cy = G.ny / 2, cz = G.nz / 2;
    std::vector<uint8_t> mask((size_t)G.nx * G.ny * G.nz, 0);
    int n_mask = 0;
    for (int k = 0; k < G.nz; ++k)
        for (int j = 0; j < G.ny; ++j)
            for (int i = 0; i < G.nx; ++i) {
                const int dx = i - cx, dy = j - cy, dz = k - cz;

                // Ellipsoidal low-k mask:
                //   (dx/kx_half)^2 + (dy/ky_half)^2 + (dz/kz_half)^2 <= 1
                // This reduces to the old spherical mask when
                // kx_half = ky_half = kz_half = K/2.
                const double rx = static_cast<double>(dx) / static_cast<double>(kx_half);
                const double ry = static_cast<double>(dy) / static_cast<double>(ky_half);
                const double rz = static_cast<double>(dz) / static_cast<double>(kz_half);
                if (rx * rx + ry * ry + rz * rz <= 1.0) {
                    mask[idx3(i, j, k, G.nx, G.ny)] = 1;
                    ++n_mask;
                }
            }
    std::cout << "Low-k ellipsoidal mask: Kx=" << Kx_mask
        << ", Ky=" << Ky_mask
        << ", Kz=" << Kz_mask
        << ", half_axes=(" << kx_half << ", " << ky_half << ", " << kz_half << ")"
        << ", active_cells=" << n_mask << std::endl;

    // --------- 5) x_base, RMS0 ----------
    std::vector<double> x_base(samples.size(), 0.0);
    for (size_t i = 0; i < invertedRhoIDToElementVector.size(); ++i)
        x_base[i] = std::log10(invertedRhoIDToElementVector[i]->resistivity);

    const double RMS0 = RunFowardCalc(x_base, true);

    std::vector<double> gd(samples.size(), 0.0), gm(samples.size(), 0.0);
    for (size_t i = 0; i < invertedRhoIDToElementVector.size(); ++i) {
        const auto* el = invertedRhoIDToElementVector[i];
        // d/d(log10 rho) = (d/d rho) * rho * ln(10)
        gd[i] = el->dDataMisfitDRho * el->resistivity / std::log10(std::exp(1));
        gm[i] = el->dRoughnessTermDRho * el->resistivity / std::log10(std::exp(1));
    }


    // --------- 6) χ²ベースのRMS帯域 ----------
    const int n_freedom = numOfObsData;
    const double chi2_lower1 = quantile(chi_squared(n_freedom), confidenceLevel1 / 2.0);
    const double chi2_lower2 = quantile(chi_squared(n_freedom), confidenceLevel2 / 2.0);
    double threshold_deltaRMS1 = std::sqrt((n_freedom * RMS0 * RMS0) / chi2_lower1) - RMS0;
    double threshold_deltaRMS2 = std::sqrt((n_freedom * RMS0 * RMS0) / chi2_lower2) - RMS0;
    if (deltaRMSLevel1 > 0.0 && deltaRMSLevel2 > 0.0) {
        threshold_deltaRMS1 = deltaRMSLevel1;
        threshold_deltaRMS2 = deltaRMSLevel2;
    }
    const double lower = RMS0 + threshold_deltaRMS1;
    const double upper = RMS0 + threshold_deltaRMS2;
    std::cout << "RMS0,RMS_lower,RMS_upper:" << RMS0 << "," << lower << "," << upper << endl;

    // --------- 7) 候補プールを作成し、多様性基準で方向を選んでから D を構築 ----------
    const int kAxes = numEnsemble;
    const uint64_t seed0 = 20150101;
    const double fd_eps = this->fd_eps;
    const double null_sv_ratio_thresh = this->null_sv_ratio_thresh;
    const int max_null_directions = this->max_null_directions;
    const int min_null_directions = this->min_null_directions;
    const int max_redraw_trials = this->max_redraw_trials;

    const double min_raw_direction_norm = 1.0e-10;

    // Window settings

    // candidate-pool selection settings
    const int candidate_pool_target = (this->candidate_pool_target > 0) ? this->candidate_pool_target : (kAxes * 3);
    const double max_allowed_selected_cosine = this->max_allowed_selected_cosine; // stop if the best next candidate is still too similar
    const bool first_direction_random = this->first_direction_random;       // false: largest norm, true: random

    std::vector<std::vector<double>> fixed_cols;
    if (orthogonalize == "gd") fixed_cols.push_back(gd);
    else if (orthogonalize == "both") {
        fixed_cols.push_back(gd);
        fixed_cols.push_back(gm);
    }
    else if (orthogonalize == "objectiveFunction") {
        std::vector<double> objfunc(samples.size(), 0.0);
        for (int ii = 0; ii < (int)samples.size(); ++ii) {
            objfunc[ii] = gd[(size_t)ii] + gm[(size_t)ii];
        }
        fixed_cols.push_back(std::move(objfunc));
    }

    std::vector<std::vector<double>> candidate_dirs;
    std::vector<double> candidate_norms;
    candidate_dirs.reserve((size_t)candidate_pool_target);
    candidate_norms.reserve((size_t)candidate_pool_target);

    for (int trial = 0; (int)candidate_dirs.size() < candidate_pool_target; ++trial) {
        std::vector<std::complex<double>> Xs = Xs_base;

        std::mt19937_64 rng(seed0 + (uint64_t)trial * 1000003ULL);

        draw_correlated_lowfreq(Xs, mask, G.nx, G.ny, G.nz,
            epsR, epsT, attenuation, corr_cells_x, corr_cells_y, corr_cells_z, rng);

        std::vector<std::complex<double>> Xu, xpropC;
        ifftshift3d(Xu, Xs, G.nx, G.ny, G.nz);
        fft3.inv(xpropC, Xu);

        std::vector<double> F_prop(xpropC.size());
        for (size_t n = 0; n < xpropC.size(); ++n) F_prop[n] = xpropC[n].real();

        std::vector<double> delta_grid(F.size());
        for (size_t n = 0; n < F.size(); ++n) {
            const double delta_raw = (F_prop[n] - F[n] * W_fft[n]);
            delta_grid[n] = delta_raw * W_out[n];
        }

        std::vector<double> delta_at_samples(samples.size());
        for (size_t s = 0; s < samples.size(); ++s) {
            double v = trilinear_sample(delta_grid, G, samples[s].x, samples[s].y, samples[s].z);
            delta_at_samples[s] = std::isfinite(v) ? v : 0.0;
        }

        const double raw_norm = l2norm(delta_at_samples);
        if (raw_norm <= min_raw_direction_norm) {
            std::cout << "reject candidate at trial " << trial
                << " because raw_norm is too small: " << raw_norm << std::endl;
            continue;
        }

        // remove only the fixed forbidden subspace (gd/gm/objective), not previously selected candidates
        std::vector<double> d_resid = orth_to_cols_qr(delta_at_samples, fixed_cols, 1e-12, false);
        const double resid_norm = l2norm(d_resid);

        normalize_inplace(d_resid);
        const double d_norm = l2norm(d_resid);
        if (d_norm <= min_raw_direction_norm) {
            std::cout << "reject candidate at trial " << trial
                << " because normalized residual norm is too small." << std::endl;
            continue;
        }

        candidate_dirs.push_back(std::move(d_resid));
        candidate_norms.push_back(d_norm);

        std::cout << "accepted candidate at trial " << trial
            << ", pool_size=" << candidate_dirs.size() << "/" << candidate_pool_target
            << std::endl;
    }

    if (candidate_dirs.empty()) {
        throw std::runtime_error("No valid FFT candidate directions were generated for D.");
    }

    std::vector<std::vector<double>> D_cols;
    D_cols.reserve((size_t)std::min(kAxes, (int)candidate_dirs.size()));
    std::vector<int> selected_candidate_indices;
    selected_candidate_indices.reserve((size_t)std::min(kAxes, (int)candidate_dirs.size()));
    std::vector<uint8_t> used((size_t)candidate_dirs.size(), 0);

    int first_idx = -1;
    if (first_direction_random) {
        std::mt19937_64 rng_first(seed0 + 987654321ULL);
        std::uniform_int_distribution<int> uid(0, (int)candidate_dirs.size() - 1);
        first_idx = uid(rng_first);
    }
    else {
        first_idx = (int)std::distance(candidate_norms.begin(),
            std::max_element(candidate_norms.begin(), candidate_norms.end()));
    }

    if (first_idx < 0 || first_idx >= (int)candidate_dirs.size()) {
        throw std::runtime_error("Failed to choose the first candidate direction.");
    }

    D_cols.push_back(candidate_dirs[(size_t)first_idx]);
    selected_candidate_indices.push_back(first_idx);
    used[(size_t)first_idx] = 1;
    std::cout << "selected first candidate index=" << first_idx
        << ", norm=" << candidate_norms[(size_t)first_idx] << std::endl;

    while ((int)D_cols.size() < kAxes) {
        int best_idx = -1;
        double best_score = std::numeric_limits<double>::infinity();

        for (int ic = 0; ic < (int)candidate_dirs.size(); ++ic) {
            if (used[(size_t)ic]) continue;

            double max_cos_to_selected = 0.0;
            for (const auto& sel : D_cols) {
                max_cos_to_selected = std::max(
                    max_cos_to_selected,
                    cosine_similarity_abs(candidate_dirs[(size_t)ic], sel)
                );
            }

            if (max_cos_to_selected < best_score) {
                best_score = max_cos_to_selected;
                best_idx = ic;
            }
        }

        if (best_idx < 0) {
            std::cout << "STOP: no unused candidate remains." << std::endl;
            break;
        }

        std::cout << "best next candidate index=" << best_idx
            << ", max cosine similarity to selected = " << best_score << std::endl;

        if (best_score >= max_allowed_selected_cosine) {
            std::cout << "STOP: best next candidate is still too similar to selected set. "
                << "best_score=" << best_score
                << ", threshold=" << max_allowed_selected_cosine << std::endl;
            break;
        }

        D_cols.push_back(candidate_dirs[(size_t)best_idx]);
        selected_candidate_indices.push_back(best_idx);
        used[(size_t)best_idx] = 1;
    }

    if (D_cols.empty()) {
        throw std::runtime_error("No selected FFT directions were available for D.");
    }

    const int nModel = (int)D_cols.front().size();
    const int kSub = (int)D_cols.size();
    cout << "D selection finished. nModel=" << nModel
        << ", nCandidate=" << candidate_dirs.size()
        << ", kSub(selected raw)=" << kSub << endl;

    // NOTE:
    // ここでは、選択済みの FFT 方向 D_cols をこの時点では QR 直交化しない。
    // 先に raw の滑らかな方向に対して有限差分で J*D_raw を作り、その後で
    // D_raw = Q R を用いて J*Q = (J*D_raw) R^{-1} に変換する。
    // これにより、QR 後の残差的・ギザギザな方向に対して forward 差分を取ることを避ける。
    write_D_cols("D_directions_raw.txt", D_cols);

    std::vector<double> base_log10rho(invertedRhoIDToElementVector.size(), 0.0);
    for (size_t i = 0; i < invertedRhoIDToElementVector.size(); ++i) {
        base_log10rho[i] = std::log10(invertedRhoIDToElementVector[i]->resistivity);
    }
    write_direction_vtk_from_elements(
        "D_directions_raw.vtk",
        invertedRhoIDToElementVector,
        D_cols,
        &base_log10rho
    );

    Eigen::MatrixXd Dmat_raw = vectors_to_column_matrix(D_cols);
    {
        Eigen::MatrixXd GramRaw = Dmat_raw.transpose() * Dmat_raw;
        cout << "Raw selected directions: ||D_raw^T D_raw - I||_F = "
            << (GramRaw - Eigen::MatrixXd::Identity(GramRaw.rows(), GramRaw.cols())).norm() << endl;
        const int nShow = std::min(kSub, 5);
        for (int i = 0; i < nShow; ++i) {
            for (int j = 0; j < nShow; ++j) {
                cout << "dot(D_raw[" << i << "],D_raw[" << j << "]) = " << GramRaw(i, j) << endl;
            }
        }
    }

    // --------- 8) 前進差分で J*D_raw を構築 ----------
    // 注意:
    // ここは各データ値ベクトルを返す関数に置き換えてください。
    // 例:
    //   std::vector<double> pred0 = RunForwardCalcPredictedData(x_base);
    // という関数を別途用意し、その関数で差し替える想定です。
    auto evalPredictedData = [&](const std::vector<double>& x_model) -> std::vector<double> {
        // TODO:
        // 以下をユーザー側で実装済みの「各データ値を返す関数」に置き換えてください。
        // 例:
        // return RunForwardCalcPredictedData(x_model);
        //throw std::runtime_error("evalPredictedData is not implemented. Replace this lambda with your data-vector forward function.");
        return RunFowardCalcForJacobian(x_model);
        };

    std::vector<double> data0 = evalPredictedData(x_base);
    if (data0.empty()) {
        throw std::runtime_error("Predicted data vector is empty.");
    }
    ensure_finite_vector_or_throw(data0, "data0");

    Eigen::MatrixXd Jstar((int)data0.size(), kSub);
    for (int j = 0; j < kSub; ++j) {
        const std::vector<double>& q_col = D_cols[(size_t)j];
        const double maxabs_q = max_abs_value(q_col);
        if (maxabs_q <= 1.0e-12) {
            throw std::runtime_error("Encountered near-zero basis column in finite difference.");
        }

        std::vector<double> d_fd(q_col.size(), 0.0);
        for (size_t i = 0; i < q_col.size(); ++i) {
            d_fd[i] = q_col[i] / maxabs_q;
        }

        std::vector<double> x_fd = x_base;
        for (size_t i = 0; i < x_fd.size(); ++i) {
            x_fd[i] += fd_eps * d_fd[i];
        }

        std::vector<double> data1 = evalPredictedData(x_fd);
        if (data1.size() != data0.size()) {
            throw std::runtime_error("Predicted data vector size mismatch in finite difference.");
        }
        ensure_finite_vector_or_throw(data1, "data1", j);

        std::vector<double> jd_col(data0.size(), 0.0);
        for (size_t i = 0; i < data0.size(); ++i) {
            const double val = (data1[i] - data0[i]) * maxabs_q / fd_eps;
            if (!std::isfinite(val)) {
                std::ostringstream oss;
                oss << std::setprecision(17)
                    << "Non-finite JD entry detected"
                    << ": column=" << j
                    << ", row=" << i
                    << ", data0=" << data0[i]
                    << ", data1=" << data1[i]
                    << ", maxabs_q=" << maxabs_q
                    << ", fd_eps=" << fd_eps
                    << ", q_col=" << q_col[i]
                    << ", d_fd=" << d_fd[i]
                    << ", x_base=" << x_base[i]
                    << ", x_fd=" << x_fd[i];
                throw std::runtime_error(oss.str());
            }
            jd_col[i] = val;
            Jstar((int)i, j) = val;
        }
        ensure_finite_vector_or_throw(jd_col, "jd_col", j);

        double jd_norm = l2norm(jd_col);
        const auto jd_minmax = minmax_value(jd_col);
        cout << "JD column " << j
            << " norm = " << jd_norm
            << " (fd maxabs scale = " << maxabs_q << ")"
            << ", min = " << jd_minmax.first
            << ", max = " << jd_minmax.second << endl;
    }

    ensure_finite_matrix_or_throw(Jstar, "Jstar_raw before post-FD QR transform");
    cout << "Jstar_raw (= J*D_raw) rows=" << Jstar.rows() << ", cols=" << Jstar.cols() << endl;
    cout << "Jstar_raw Frobenius norm=" << Jstar.norm() << endl;
    cout << "Jstar_raw minCoeff=" << Jstar.minCoeff() << ", maxCoeff=" << Jstar.maxCoeff() << endl;

    // --------- 8.5) 有限差分後に D_raw を QR 直交化し、J*D_raw も同じ基底へ変換 ----------
    // D_raw = Q R なので、J*D_raw = (J*Q) R。
    // よって J*Q = (J*D_raw) R^{-1} を作り、以降は直交基底 Q と J*Q で SVD する。
    {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Dmat_raw);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(Dmat_raw.rows(), Dmat_raw.cols());
        Eigen::MatrixXd R = qr.matrixQR()
            .topLeftCorner(Dmat_raw.cols(), Dmat_raw.cols())
            .template triangularView<Eigen::Upper>();

        // Solve (JQ) * R = Jraw  <=>  R^T * (JQ)^T = Jraw^T
        Eigen::MatrixXd Jstar_orth_T = R.transpose()
            .template triangularView<Eigen::Lower>()
            .solve(Jstar.transpose());
        Jstar = Jstar_orth_T.transpose();
        D_cols = column_matrix_to_vectors(Q);

        write_D_cols("D_directions.txt", D_cols);
        write_direction_vtk_from_elements(
            "D_directions.vtk",
            invertedRhoIDToElementVector,
            D_cols,
            &base_log10rho
        );

        Eigen::MatrixXd Gram = Q.transpose() * Q;
        cout << "Post-FD QR: ||Q^T Q - I||_F = "
            << (Gram - Eigen::MatrixXd::Identity(Gram.rows(), Gram.cols())).norm() << endl;
        cout << "Post-FD QR: R minCoeff=" << R.minCoeff()
            << ", maxCoeff=" << R.maxCoeff()
            << ", abs(det(R))=" << std::abs(R.determinant()) << endl;
        const int nShow = std::min(kSub, 5);
        for (int i = 0; i < nShow; ++i) {
            for (int j = 0; j < nShow; ++j) {
                cout << "dot(Q[" << i << "],Q[" << j << "]) = " << Gram(i, j) << endl;
            }
        }
    }

    ensure_finite_matrix_or_throw(Jstar, "Jstar after post-FD QR transform");
    cout << "Jstar (= J*Q after post-FD QR) rows=" << Jstar.rows() << ", cols=" << Jstar.cols() << endl;
    cout << "Jstar Frobenius norm=" << Jstar.norm() << endl;
    cout << "Jstar minCoeff=" << Jstar.minCoeff() << ", maxCoeff=" << Jstar.maxCoeff() << endl;
    {
        Eigen::MatrixXd Gjtj = Jstar.transpose() * Jstar;
        cout << "||Jstar^T Jstar - I||_F = "
            << (Gjtj - Eigen::MatrixXd::Identity(Gjtj.rows(), Gjtj.cols())).norm() << endl;
        const int nShow = std::min(kSub, 5);
        for (int i = 0; i < nShow; ++i) {
            for (int j = 0; j < nShow; ++j) {
                cout << "(J^T J)[" << i << "," << j << "] = " << Gjtj(i, j) << endl;
            }
        }
    }

    // --------- 9) J* の SVD ----------
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jstar, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singVals = svd.singularValues();
    const Eigen::MatrixXd V = svd.matrixV();

    {
        std::ofstream ofs("projected_jacobian_singular_values.txt");
        ofs << "# idx singular_value\n";
        for (int i = 0; i < singVals.size(); ++i) {
            ofs << i << " " << singVals(i) << "\n";
        }
    }

    // --------- 10) 小さい特異値方向を δm = D v でモデル空間へ戻す ----------
    // JacobiSVD は特異値を大きい順に返すので、後ろから候補化する。
    struct NullDirectionResult {
        int svd_idx;
        double singular_value;
        double singular_value_ratio;
        double a_neg;
        double a_pos;
        std::vector<double> dir;
    };
    std::vector<NullDirectionResult> results;
    results.reserve((size_t)kSub);

    std::vector<int> selected_svd_indices;
    selected_svd_indices.reserve((size_t)std::min(kSub, max_null_directions));

    const double sigma_max = (singVals.size() > 0 ? singVals(0) : 0.0);
    const double sigma_floor = 1.0e-300;

    cout << std::setprecision(17);
    cout << "Singular values of projected Jacobian:" << endl;
    for (int i = 0; i < singVals.size(); ++i) {
        const double ratio = (sigma_max > sigma_floor ? singVals(i) / sigma_max : 0.0);
        cout << "  sv[" << i << "] = " << singVals(i) << ", ratio = " << ratio << endl;
    }
    cout << std::setprecision(6);

    // まず相対特異値しきい値で候補選択（小さい順）
    for (int rank_from_small = 0; rank_from_small < kSub; ++rank_from_small) {
        const int svd_idx = kSub - 1 - rank_from_small;
        const double sigma = singVals(svd_idx);
        const double ratio = (sigma_max > sigma_floor ? sigma / sigma_max : 0.0);
        if (ratio <= null_sv_ratio_thresh) {
            selected_svd_indices.push_back(svd_idx);
        }
    }

    // 候補が少なすぎる場合は、小さい方から最低 min_null_directions 本まで補う
    for (int rank_from_small = 0;
        (int)selected_svd_indices.size() < std::min(kSub, min_null_directions) && rank_from_small < kSub;
        ++rank_from_small) {
        const int svd_idx = kSub - 1 - rank_from_small;
        if (std::find(selected_svd_indices.begin(), selected_svd_indices.end(), svd_idx) == selected_svd_indices.end()) {
            selected_svd_indices.push_back(svd_idx);
        }
    }

    // 多すぎる場合は小さい特異値側から max_null_directions 本に切る
    if ((int)selected_svd_indices.size() > max_null_directions) {
        selected_svd_indices.resize((size_t)max_null_directions);
    }

    cout << "Selected null-direction count = " << selected_svd_indices.size()
        << " (threshold ratio=" << null_sv_ratio_thresh
        << ", min=" << min_null_directions
        << ", max=" << max_null_directions << ")" << endl;

    for (size_t sel_rank = 0; sel_rank < selected_svd_indices.size(); ++sel_rank) {
        const int svd_idx = selected_svd_indices[sel_rank];
        Eigen::VectorXd coeff = V.col(svd_idx);
        cout << std::setprecision(17);
        cout << "selected null direction " << sel_rank
            << ": svd_idx=" << svd_idx
            << ", coeff.norm=" << coeff.norm()
            << ", coeff.min=" << coeff.minCoeff()
            << ", coeff.max=" << coeff.maxCoeff() << endl;

        std::vector<double> d_null = linear_combination_from_columns(D_cols, coeff);
        const double nrm_d_null = l2norm(d_null);
        const double maxabs_d_null = max_abs_value(d_null);
        const auto dnull_minmax = minmax_value(d_null);
        cout << "selected null direction " << sel_rank
            << ": reconstructed d_null norm=" << nrm_d_null
            << ", maxabs=" << maxabs_d_null
            << ", min=" << dnull_minmax.first
            << ", max=" << dnull_minmax.second << endl;
        cout << std::setprecision(6);
        if (nrm_d_null <= 1e-12) {
            cout << "skip selected null direction " << sel_rank << " because L2 norm is too small." << endl;
            continue;
        }
        for (double& v : d_null) {
            v /= nrm_d_null;
        }

        EvalRMS evalRMS = [&](double a) {
            std::vector<double> x_trial(x_base.size());
            for (size_t i = 0; i < x_trial.size(); ++i) x_trial[i] = x_base[i] + a * d_null[i];
            return RunFowardCalc(x_trial, false);
            };

        double a_pos = secant_one_side(+1.0, evalRMS, RMS0, lower, upper);
        double a_neg = secant_one_side(-1.0, evalRMS, RMS0, lower, upper);

        const double sigma = singVals(svd_idx);
        const double ratio = (sigma_max > sigma_floor ? sigma / sigma_max : 0.0);

        results.push_back({ svd_idx, sigma, ratio, a_neg, a_pos, d_null });

        cout << "selected-null-dir rank=" << sel_rank
            << ", svd_idx=" << svd_idx
            << ", sigma=" << sigma
            << ", ratio=" << ratio
            << ", a_neg=" << a_neg
            << ", a_pos=" << a_pos << endl;

        std::string filenameOut = "NullAxis" + std::to_string((int)sel_rank) + "_pos.vtk";
        Eigen::VectorXd tmp((int)d_null.size());
        for (int ii = 0; ii < (int)d_null.size(); ++ii) {
            tmp.coeffRef(ii) = a_pos * d_null[(size_t)ii];
        }
        output->VTKFileOputput(&invertedRhoIDToElementVector, &tmp, filenameOut);

        filenameOut = "NullAxis" + std::to_string((int)sel_rank) + "_neg.vtk";
        tmp.setZero();
        for (int ii = 0; ii < (int)d_null.size(); ++ii) {
            tmp.coeffRef(ii) = a_neg * d_null[(size_t)ii];
        }
        output->VTKFileOputput(&invertedRhoIDToElementVector, &tmp, filenameOut);
    }

    // --------- 11) サマリ出力 ----------
    {
        std::ofstream ofs("projected_null_axes_summary.txt");
        ofs << "# null_sv_ratio_thresh " << null_sv_ratio_thresh << "\n";
        ofs << "# min_null_directions " << min_null_directions << "\n";
        ofs << "# max_null_directions " << max_null_directions << "\n";
        ofs << "# selected_rank svd_idx singular_value singular_value_ratio a_neg a_pos\n";
        for (size_t i = 0; i < results.size(); ++i) {
            ofs << i << " "
                << results[i].svd_idx << " "
                << results[i].singular_value << " "
                << results[i].singular_value_ratio << " "
                << results[i].a_neg << " "
                << results[i].a_pos << "\n";
        }
    }

    std::cout << "Projected Jacobian / null-space analysis finished. (kSub=" << kSub << ")\n";
    return;
}


double Analysis::Analysis::RunFowardCalc(std::vector<double> x, bool isCalcGradient) {
    vector<double> resis_pre(invertedRhoIDToElementVector.size(), 0.0);
    for (int i = 0; i < resis_pre.size(); i++) {
        resis_pre[i] = invertedRhoIDToElementVector[i]->resistivity;
        invertedRhoIDToElementVector[i]->resistivity = std::pow(10.0, x[i]);
    }
    if (usePreviousResult && resultVector_init.size() != 0) {
        resultVector = resultVector_init;
    }
    else {
        resultVector.setZero();
    }

    SetSameResistivityToBoundaryCell();
    CalcSurfaceResistivityElements(); //Update Resistivity
    CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
    CalcForward(isCalcGradient);
    if (isCalcGradient) {

        CalcDDataMisfitDRho();
        CalcDJDRho();  //here, gradient terms are calculated
    }
    double dData = CalcDataMisfit();
    double rms = std::pow(dData / numOfObsData, 0.5);
    double maxChangeOrder = 0.0;
    for (int i = 0; i < resis_pre.size(); i++) {
        if (maxChangeOrder < abs(std::log10(invertedRhoIDToElementVector[i]->resistivity / resis_pre[i]))) {
            maxChangeOrder = abs(std::log10(invertedRhoIDToElementVector[i]->resistivity / resis_pre[i]));
        }
    }
    cout << "maxChangeOrder:" << maxChangeOrder << endl;
    for (int i = 0; i < resis_pre.size(); i++) {
        invertedRhoIDToElementVector[i]->resistivity = resis_pre[i];
    }
    if (usePreviousResult && resultVector_init.size() == 0) {
        resultVector_init = resultVector;
    }

    return rms;

}

vector<double> Analysis::Analysis::RunFowardCalcForJacobian(std::vector<double> x) {
    vector<double> resis_pre(invertedRhoIDToElementVector.size(), 0.0);
    for (int i = 0; i < resis_pre.size(); i++) {
        resis_pre[i] = invertedRhoIDToElementVector[i]->resistivity;
        invertedRhoIDToElementVector[i]->resistivity = std::pow(10.0, x[i]);
    }
    if (usePreviousResult && resultVector_init.size() != 0) {
        resultVector = resultVector_init;
    }
    else {
        resultVector.setZero();
    }
    isFirstLambdaAndLoop = false; // this is needed to restart iterative solver when it is not converged.

    SetSameResistivityToBoundaryCell();
    CalcSurfaceResistivityElements(); //Update Resistivity
    CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
    FFTSensitivityMode = true;
    bool isCalcGradient = false;
    CalcForward(isCalcGradient);

    vector<double> returnVec;
    returnVec.reserve(numOfObsData);
    returnVec = CalcDataMisfitEachData();

    double maxChangeOrder = 0.0;
    for (int i = 0; i < resis_pre.size(); i++) {
        if (maxChangeOrder < abs(std::log10(invertedRhoIDToElementVector[i]->resistivity / resis_pre[i]))) {
            maxChangeOrder = abs(std::log10(invertedRhoIDToElementVector[i]->resistivity / resis_pre[i]));
        }
    }
    cout << "maxChangeOrder:" << maxChangeOrder << endl;
    for (int i = 0; i < resis_pre.size(); i++) {
        invertedRhoIDToElementVector[i]->resistivity = resis_pre[i];
    }
    if (usePreviousResult && resultVector_init.size() == 0) {
        resultVector_init = resultVector;
    }

    return returnVec;

}

vector<double> Analysis::Analysis::CalcDataMisfitEachData() {
    dataMisfit = 0.0;

    vector<double> dataMisfitVec;
    dataMisfitVec.reserve(numOfObsData);
    //Impedance Tensor
    for (int i = 0; i < numOfObsPointElements; i++) {
        Element::Element* element = obsPointElements[i];
        if (element->isInversionImpedance == true) {
            for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
                Eigen::Matrix2cd Zcalc;


                if (isInvertedDistortion) {
                    Zcalc = element->distortionMatrix * element->Z[iOmega];

                }
                else {
                    Zcalc = element->Z[iOmega];

                }
                for (int ii = 0; ii < 2; ii++) {
                    for (int jj = 0; jj < 2; jj++) {
                        std::complex<double> dZtmp = Zcalc.coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
                        double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
                        double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

                        if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0) {
                            dZtmp.real(dZtmp.real() / epsReal);
                        }
                        else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
                            dZtmp.real(0.0);
                        }
                        else {
                            //そのまま
                        }
                        if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0) {
                            dZtmp.imag(dZtmp.imag() / epsImag);
                        }
                        else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
                            dZtmp.imag(0.0);
                        }
                        else {
                            //そのまま
                        }

                        double misfitReal = dZtmp.real();
                        double misfitImag = dZtmp.imag();
                        dataMisfitVec.push_back(misfitReal);
                        dataMisfitVec.push_back(misfitImag);



                    }
                }
            }
        }
    }

    //Tipper 
    for (int i = 0; i < numOfObsPointElements; i++) {
        Element::Element* element = obsPointElements[i];
        if (element->isInversionTipper == true) {
            for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
                for (int ii = 0; ii < 2; ii++) {
                    std::complex<double> dTtmp = element->T[iOmega].coeff(ii) - element->tipperObsData->TobsVector[iOmega].coeff(ii);
                    double epsReal = std::abs(element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii));
                    double epsImag = std::abs(element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii));

                    if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) > 0) {
                        dTtmp.real(dTtmp.real() / epsReal);
                    }
                    else if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) <= 0) {
                        dTtmp.real(0.0);
                    }
                    else {
                        //そのまま
                    }
                    if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) > 0) {
                        dTtmp.imag(dTtmp.imag() / epsImag);
                    }
                    else if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) <= 0) {
                        dTtmp.imag(0.0);
                    }
                    else {
                        //そのまま
                    }

                    double misfitReal = dTtmp.real();
                    double misfitImag = dTtmp.imag();
                    dataMisfitVec.push_back(misfitReal);
                    dataMisfitVec.push_back(misfitImag);


                }

            }
        }
    }
    // Todo::他のテンソル量を逆解析する場合はここに足す
    //dataMisfit /= numOfObsData;


    return dataMisfitVec;
}

