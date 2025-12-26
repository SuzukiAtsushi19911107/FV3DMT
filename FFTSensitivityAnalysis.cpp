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
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace {
    // ========================= ユーティリティ =========================
    static inline int _sym_idx(int i, int N, int c) { return (2 * c - i + 2 * N) % N; }

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
    static inline void make_cosine_window_on_grid(std::vector<double>& W,
        int NX, int NY, int NZ, int t_cells = 3, double eps = 0.05) {
        W.assign((size_t)NX * NY * NZ, 1.0);
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    int dx = std::min(i, NX - 1 - i);
                    int dy = std::min(j, NY - 1 - j);
                    int dz = std::min(k, NZ - 1 - k);
                    int d = std::min(dx, std::min(dy, dz));
                    double w = 1.0;
                    if (d == 0) w = 0.0;
                    else if (d < t_cells) {
                        w = 0.5 * (1.0 - std::cos(M_PI * (double)d / (double)t_cells));
                    }
                    W[idx3(i, j, k, NX, NY)] = eps + (1.0 - eps) * w;
                }
    }

    // ========================= 低周波微摂動（均一ロールオフ） =========================
    static inline void draw_uniform_lowfreq(
        std::vector<std::complex<double>>& Fh, // in/out: fftshift 後
        const std::vector<uint8_t>& mask,      // bool相当 (1=更新)
        int NX, int NY, int NZ,
        double epsR, double epst, double attenuation,
        std::mt19937_64& rng)
    {
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        int cx = NX / 2, cy = NY / 2, cz = NZ / 2;

        // maskの有効max m を推定
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
        double mxmax = std::max(1, mmax_of(0));
        double mymax = std::max(1, mmax_of(1));
        double mzmax = std::max(1, mmax_of(2));

        std::vector<double> w(NX * NY * NZ, 0.0);
        double att = std::clamp(attenuation, 1e-6, 1.0);
        double gamma = std::sqrt(-std::log(att));
        for (int k = 0; k < NZ; ++k) {
            double rz = (k - cz) / mzmax;
            for (int j = 0; j < NY; ++j) {
                double ry = (j - cy) / mymax;
                for (int i = 0; i < NX; ++i) {
                    double rx = (i - cx) / mxmax;
                    double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                    double ww = std::exp(-(gamma * std::clamp(r, 0.0, 1.0)) * (gamma * std::clamp(r, 0.0, 1.0)));
                    if (r > 1.0) ww = 0.0;
                    w[idx3(i, j, k, NX, NY)] = ww;
                }
            }
        }

        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    auto id = idx3(i, j, k, NX, NY);
                    if (!mask[id]) continue;
                    if (i == cx && j == cy && k == cz) continue;
                    int jx = _sym_idx(i, NX, cx), jy = _sym_idx(j, NY, cy), jz = _sym_idx(k, NZ, cz);
                    // 代表側のみ更新
                    if ((i > jx) || (i == jx && j > jy) || (i == jx && j == jy && k > jz)) continue;

                    double ww = w[id]; if (ww <= 0) continue;

                    // 振幅±epsR, 位相±epst
                    double a_rand = 1.0 + epsR * U(rng);
                    double th_rand = epst * U(rng);
                    std::complex<double> alpha = std::polar(a_rand * ww, th_rand);

                    auto Fk_old = Fh[id];
                    //cout <<id<<" "<< a_rand << " " << alpha << " " << Fk_old << " " << ww<<endl;
                    auto Fk_new = Fk_old * alpha;
                    if (i == jx && j == jy && k == jz) {
                        Fh[id] = std::complex<double>(Fk_new.real(), 0.0);
                    }
                    else {
                        Fh[id] = Fk_new;
                        Fh[idx3(jx, jy, jz, NX, NY)] = std::conj(Fk_new);
                    }
                }
        // DC維持
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
        double fx = (x - g.ox) / g.dx, fy = (y - g.oy) / g.dy, fz = (z - g.oz) / g.dz;
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
        return 0;
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
}
void Analysis::Analysis::RunFFTSensitivityAnalysis() {

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
    std::vector<Pt> samples;
    samples.reserve(invertedRhoIDToElementVector.size());
    for (size_t i = 0; i < invertedRhoIDToElementVector.size(); ++i) {
        const auto* el = invertedRhoIDToElementVector[i];
        const int idz = std::clamp(el->IDZ, 0, (int)averageZs.size() - 1);
        const double x = el->centerCoord.coeff(0);
        const double y = el->centerCoord.coeff(1);
        const double z = averageZs[idz];
        const double v = std::log10(el->resistivity);
        //cout << "V" <<x<<" "<<y<<" "<<z<< " " << v << endl;
        samples.push_back({ x,y,z,v });
    }

    // --------- 3) 規則格子（FFT用） ----------
    GridSpec G{
        Nx, Ny, Nz,
        minX, minY, minZ,
        (maxX - minX) / double(Nx), (maxY - minY) / double(Ny), (maxZ - minZ) / double(Nz)
    };

    std::vector<double> F;
    //sample_nearest_to_grid(samples, G, F);
    sample_idw_to_grid(samples, G, F);

    // NaN→平均で埋め & 平均差し引き（DC抑制）
    double mean = 0.0; int cnt = 0;
    for (double v : F) { if (std::isfinite(v)) { mean += v; ++cnt; } }
    mean = (cnt > 0 ? mean / cnt : 0.0);
    for (double& v : F) { if (!std::isfinite(v)) v = mean; v -= mean; }

    // --------- 4) FFT & マスク & 窓（固定前処理） ----------
    std::vector<double> W;
    make_cosine_window_on_grid(W, G.nx, G.ny, G.nz, /*t_cells=*/cells_window, /*eps=*/eps_window);

    FFT3D fft3(G.nx, G.ny, G.nz);
    std::vector<std::complex<double>> x(G.nx * G.ny * G.nz), X, Xs_base;
    for (size_t n = 0; n < F.size(); ++n) x[n] = std::complex<double>(F[n], 0.0)*W[n];

    double norm = 0.0;
    for (int i = 0; i < x.size(); i++) {
        norm += (x[i] * std::conj(x[i])).real();
    }
    cout << "before fft3_X:" << norm << endl;
    fft3.fwd(X, x);
    fftshift3d(Xs_base, X, G.nx, G.ny, G.nz);
    norm = 0.0;
    for (int i = 0; i < Xs_base.size(); i++) {
        norm += (Xs_base[i] * std::conj(Xs_base[i])).real();
    }
    cout << "after fftshift3d:" << norm << endl;
    const int k_half = K / 2;
    const int cx = G.nx / 2, cy = G.ny / 2, cz = G.nz / 2;
    std::vector<uint8_t> mask(G.nx * G.ny * G.nz, 0);
    for (int k = 0; k < G.nz; ++k)
        for (int j = 0; j < G.ny; ++j)
            for (int i = 0; i < G.nx; ++i) {
                int dx = i - cx, dy = j - cy, dz = k - cz;
                if (dx * dx + dy * dy + dz * dz <= k_half * k_half)
                    mask[idx3(i, j, k, G.nx, G.ny)] = 1;
            }

    

    // --------- 5) x_base, 勾配, RMS0 ----------
    

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
    const double threshold_deltaRMS1 = std::sqrt((n_freedom * RMS0 * RMS0) / chi2_lower1) - RMS0;
    const double threshold_deltaRMS2 = std::sqrt((n_freedom * RMS0 * RMS0) / chi2_lower2) - RMS0;
    const double lower = RMS0 + threshold_deltaRMS1;
    const double upper = RMS0 + threshold_deltaRMS2;
    std::cout << "RMS0,RMS_lower,RMS_upper:" << RMS0 << "," << lower << "," << upper << endl;
    // --------- 7) アンサンブル反復 ----------
    const int kAxes = numEnsemble;      // ← クラス側に持っている想定（なければ定数でOK）
    const uint64_t seed0 = 20150101; // ← ベースシード（無ければ固定値）

    struct AxisResult { int idx; double a_neg, a_pos; std::vector<double> dir; };
    std::vector<AxisResult> results; results.reserve(kAxes);

    // （任意）まとめVTK用
    std::map<std::string, std::vector<double>> vtk_arrays;

    std::vector<HexCell> hexes;
    
    int count = invertedRhoIDToElementVector.size();

    int vtkID = 0;
    unordered_map<int, int> nodesIDtoVTKID;
    vector<Eigen::Vector3d> nodes;
    nodes.reserve(8 * count);
    for (int i = 0; i < count; i++) {
        Eigen::VectorXd x( 8 );
        Eigen::VectorXd y( 8 );
        Eigen::VectorXd z( 8 );
        for (int j = 0; j < 8; j++) {
            if (nodesIDtoVTKID.find(invertedRhoIDToElementVector[i]->nodes[j]->ID) == nodesIDtoVTKID.end()) {
                nodesIDtoVTKID[invertedRhoIDToElementVector[i]->nodes[j]->ID] = vtkID;
                nodes.push_back(invertedRhoIDToElementVector[i]->nodes[j]->x);
                vtkID++;
            }
        }
    }
    std::vector<Eigen::Vector3d> pts_out( nodes.size() );
    for (int i = 0; i < nodes.size(); i++) {
        pts_out[i] = nodes[i];
    }
    for (int i = 0; i < count; i++) {
        std::array<int, 8> conn;
        for (int j = 0; j < 8; j++) {
            conn[j] = nodesIDtoVTKID[invertedRhoIDToElementVector[i]->nodes[j]->ID];

        }
        hexes.push_back(HexCell{ conn });
    }
        


    for (int j = 0; j < kAxes; ++j) {
        // --- ランダム摂動（スペクトル空間） ---
        std::vector<std::complex<double>> Xs = Xs_base;
        std::mt19937_64 rng(seed0 + (uint64_t)j* seed0);
        norm = 0.0;
        for (int i = 0; i < Xs.size(); i++) {
            norm += (Xs[i] * std::conj(Xs[i])).real();
        }
        cout << "before draw_uniform_lowfreq:" << norm << endl;
        draw_uniform_lowfreq(Xs, mask, G.nx, G.ny, G.nz, epsR, epsT, attenuation, rng);
        norm = 0.0;
        for (int i = 0; i < Xs.size(); i++) {
            norm += (Xs[i] * std::conj(Xs[i])).real();
        }
        cout << "before InvFFT:" << norm << endl;
        // 逆FFT
        //const double Ntot = double(G.nx) * G.ny * G.nz;
        std::vector<std::complex<double>> Xu, xpropC;
        ifftshift3d(Xu, Xs, G.nx, G.ny, G.nz);
        fft3.inv(xpropC, Xu);
        std::vector<double> F_prop(xpropC.size());
        for (size_t n = 0; n < xpropC.size(); ++n) F_prop[n] = xpropC[n].real();
        norm = 0.0;
        for (int i = 0; i < F_prop.size(); i++) {
            norm += F_prop[i] * F_prop[i];
        }
        cout << "after InvFFT:" << norm << endl;
        // 差分 + 窓
        std::vector<double> delta_grid(F.size());
        for (size_t n = 0; n < F.size(); ++n) delta_grid[n] = (F_prop[n] - F[n]*W[n]) * W[n];
        norm = 0.0;
        for (int i = 0; i < delta_grid.size(); i++) {
            norm += delta_grid[i] * delta_grid[i];
        }
        // 再ローパス
        //lowpass_again(delta_grid, G.nx, G.ny, G.nz, k_half, k_half, k_half, attenuation);
        norm = 0.0;
        for (int i = 0; i < delta_grid.size(); i++) {
            norm += delta_grid[i] * delta_grid[i];
        }
        cout << "Before Trilinear:" << norm << endl;
        // セル中心へサンプリング
        std::vector<double> delta_at_samples(samples.size());
        for (size_t s = 0; s < samples.size(); ++s) {
            double v = trilinear_sample(delta_grid, G, samples[s].x, samples[s].y, samples[s].z);
            
            delta_at_samples[s] = std::isfinite(v) ? v : 0.0;
        }

        // g_d と g_m へ直交化
        std::vector<double> g_obj(samples.size(), 0.0);
        std::vector<std::vector<double>> cols;
        if (orthogonalize == "gd") {
            cols.push_back(gd);
        }
        else if (orthogonalize == "both") {
            cols.push_back(gd);
            cols.push_back(gm);
        }
        else {
            for (int i = 0; i < g_obj.size(); i++) {
                g_obj[i] = gd[i] + gm[i];
            }
            cols.push_back(g_obj);
        }
        norm = 0.0;
        for (int i = 0; i < delta_at_samples.size(); i++) {
            norm += delta_at_samples[i] * delta_at_samples[i];
        }
        cout <<"Before orth:"<< norm << endl;
        std::vector<double> d = orth_to_cols_qr(delta_at_samples, cols, 1e-12, false);
        norm = 0.0;
        for (int i = 0; i < d.size(); i++) {
            norm += d[i] * d[i];
        }
        cout << "AFTER orth:" << norm << endl;
        double maxd = 0.0;
        for (int i = 0; i < d.size(); i++) {
            if (maxd < std::abs(d[i])) {
                maxd = std::abs(d[i]);
            }
        }
        for (int i = 0; i < d.size(); i++) {
            d[i] = initWidth * d[i] / maxd;
        }
        // ラインサーチ（±）
        EvalRMS evalRMS = [&](double a) {
            std::vector<double> x(x_base.size());
            for (size_t i = 0; i < x.size(); ++i) x[i] = x_base[i] + a * d[i];
            return RunFowardCalc(x,false);
            };
        double a_pos = secant_one_side(+1.0, evalRMS, RMS0, lower, upper);
        double a_neg = secant_one_side(-1.0, evalRMS, RMS0, lower, upper);

        norm = 0.0;
        for (int i = 0; i < d.size(); i++) {
            norm +=  d[i]* d[i];
        }
        cout <<"After Line Search:"<< norm << endl;
        norm = std::sqrt(norm);
        a_pos = a_pos * norm;
        a_neg = a_neg * norm;
        for (int i = 0; i < d.size(); i++) {
            d[i] = d[i] / norm;
        }
        results.push_back({ j, a_neg, a_pos, d });
        // ----- ここから「各ループごとに VTK 出力」 ----
        // その回の3配列だけを詰める
        string filenameOut = "Axis" + to_string(j) + "_pos.vtk";
        Eigen::VectorXd tmp;
        tmp.resize(d.size());
        for (int ii = 0; ii < d.size(); ii++) {
            tmp.coeffRef(ii) = a_pos * d[ii];
        }
        output->VTKFileOputput(&invertedRhoIDToElementVector, &tmp, filenameOut);
        filenameOut = "Axis" + to_string(j) + "_neg.vtk";
        tmp.setZero();
        for (int ii = 0; ii < d.size(); ii++) {
            tmp.coeffRef(ii) = a_neg * d[ii];
        }
        output->VTKFileOputput(&invertedRhoIDToElementVector, &tmp, filenameOut);
        std::vector<double> lim_pos(d.size()), lim_neg(d.size());
        for (size_t i = 0; i < d.size(); ++i) {
            lim_pos[i] = a_pos * d[i];   // +側の到達上限（Δlog10ρ）
            lim_neg[i] = a_neg * d[i];   // -側の到達下限（Δlog10ρ）
        }
        //std::map<std::string, std::vector<double>> vtk_arrays_j;
        //vtk_arrays_j["dir"] = d;        // 方向そのもの（単位はΔlog10ρ）
        //vtk_arrays_j["limit_pos"] = lim_pos;  // +側の変化量
        //vtk_arrays_j["limit_neg"] = lim_neg;  // -側の変化量

        //// もし「実際のモデル値」を同梱したいなら（任意）
        //// x_plus = x_base + lim_pos, x_minus = x_base + lim_neg を追加
        //std::vector<double> x_plus(x_base.size()), x_minus(x_base.size());
        //for (size_t i = 0; i < x_base.size(); ++i) {
        //    x_plus[i] = x_base[i] + lim_pos[i];  // log10ρ（+側）
        //    x_minus[i] = x_base[i] + lim_neg[i];  // log10ρ（-側）
        //}
        //vtk_arrays_j["x_plus_log10rho"] = std::move(x_plus);
        //vtk_arrays_j["x_minus_log10rho"] = std::move(x_minus);

        //// 連番ファイル名
        //char fname[64];
        //std::snprintf(fname, sizeof(fname), "axis_%03d.vtk", j);

        //// 実メッシュで書き出し（allPts: std::vector<Eigen::Vector3d>,
        //// allHexes: std::vector<HexCell> を想定）
        //write_legacy_ugrid_vtk(fname, pts_out, hexes, vtk_arrays_j);

    }

    // --------- 8) サマリ出力（任意） ----------
    {
        std::ofstream ofs("sequential_axes_summary.txt");
        ofs << "# idx  a_neg  a_pos\n";
        for (auto& r : results) ofs << r.idx << " " << r.a_neg << " " << r.a_pos << "\n";
    }




    std::cout << "Ensemble finished. (" << kAxes << " samples)\n";
    return;
}


double Analysis::Analysis::RunFowardCalc(std::vector<double> x, bool isCalcGradient) {
    vector<double> resis_pre(invertedRhoIDToElementVector.size(), 0.0);
    for (int i = 0; i < resis_pre.size(); i++) {
        resis_pre[i] = invertedRhoIDToElementVector[i]->resistivity;
        invertedRhoIDToElementVector[i]->resistivity = std::pow(10.0,x[i]);
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
    if(isCalcGradient){

        CalcDDataMisfitDRho();
        CalcDJDRho();  //here, gradient terms are calculated
     }
    double dData=CalcDataMisfit();
    double rms=std::pow(dData / numOfObsData, 0.5);
    double maxChangeOrder = 0.0;
    for (int i = 0; i < resis_pre.size(); i++) {
        if (maxChangeOrder < abs(std::log10(invertedRhoIDToElementVector[i]->resistivity / resis_pre[i]))) {
            maxChangeOrder = abs(std::log10(invertedRhoIDToElementVector[i]->resistivity / resis_pre[i]));
        }
    }
    cout <<"maxChangeOrder:"<< maxChangeOrder << endl;
    for (int i = 0; i < resis_pre.size(); i++) {
        invertedRhoIDToElementVector[i]->resistivity = resis_pre[i];
    }
    if (usePreviousResult && resultVector_init.size() == 0) {
        resultVector_init = resultVector;
    }

    return rms;

}