#include "ba.h"
#include <Eigen/Dense>
#include <cmath>
#include <cassert>

// ── Rodrigues ─────────────────────────────────────────────────────────────────
// Compute R = Rodrigues(r) and optionally dR/dr_i for i in {0,1,2}.
static Eigen::Matrix3d rodrigues(const Eigen::Vector3d& r,
                                  Eigen::Matrix3d dR[3] = nullptr)
{
    double theta = r.norm();

    if (theta < 1e-10) {
        if (dR) {
            // R ≈ I + [r]×  →  dR/dr_i = [e_i]×
            dR[0] <<  0, 0, 0,   0, 0,-1,   0, 1, 0;
            dR[1] <<  0, 0, 1,   0, 0, 0,  -1, 0, 0;
            dR[2] <<  0,-1, 0,   1, 0, 0,   0, 0, 0;
        }
        return Eigen::Matrix3d::Identity() +
               (Eigen::Matrix3d() <<  0,-r[2],r[1],  r[2],0,-r[0],  -r[1],r[0],0).finished();
    }

    double c = std::cos(theta), s = std::sin(theta), ic = 1.0 - c;
    Eigen::Vector3d k = r / theta;
    double kx=k[0], ky=k[1], kz=k[2];

    Eigen::Matrix3d cross_k;
    cross_k << 0,-kz, ky,
               kz, 0,-kx,
              -ky, kx, 0;

    Eigen::Matrix3d R = c * Eigen::Matrix3d::Identity()
                       + ic * (k * k.transpose())
                       +  s * cross_k;

    if (!dR) return R;

    // dR/d(theta) = -s*I + s*k*k^T + c*[k]×
    Eigen::Matrix3d dRdt = -s * Eigen::Matrix3d::Identity()
                          +  s * (k * k.transpose())
                          +  c * cross_k;

    double inv_theta = 1.0 / theta;

    for (int i = 0; i < 3; ++i) {
        // d(theta)/dr_i = k_i
        // dk/dr_i       = (e_i - k_i*k) / theta
        Eigen::Vector3d ei = Eigen::Vector3d::Zero(); ei[i] = 1.0;
        Eigen::Vector3d dkdri = (ei - k[i]*k) * inv_theta;

        // dR/dr_i = k_i * dR/d(theta)  +  sum_j dR/dk_j * dk_j/dr_i
        Eigen::Matrix3d term = k[i] * dRdt;

        for (int j = 0; j < 3; ++j) {
            double w = dkdri[j];
            if (std::abs(w) < 1e-15) continue;

            Eigen::Vector3d ej = Eigen::Vector3d::Zero(); ej[j] = 1.0;
            // [e_j]×
            Eigen::Matrix3d cross_ej = Eigen::Matrix3d::Zero();
            if (j == 0) { cross_ej(1,2)=-1; cross_ej(2,1)= 1; }
            if (j == 1) { cross_ej(0,2)= 1; cross_ej(2,0)=-1; }
            if (j == 2) { cross_ej(0,1)=-1; cross_ej(1,0)= 1; }

            // dR/dk_j = ic*(e_j*k^T + k*e_j^T) + s*[e_j]×
            Eigen::Matrix3d dRdkj = ic * (ej * k.transpose() + k * ej.transpose())
                                   +  s * cross_ej;
            term += w * dRdkj;
        }
        dR[i] = term;
    }
    return R;
}

// ── Projection + analytic Jacobian ───────────────────────────────────────────
// Residual r = [u_proj - u_obs, v_proj - v_obs]
// J_cam (2×6): d(r)/d[rvec(3)|t(3)]
// J_pt  (2×3): d(r)/dX
static Eigen::Vector2d project_residual(
    const Eigen::Vector3d& rvec,
    const Eigen::Vector3d& t,
    const Eigen::Vector3d& X,
    double fx, double fy, double cx, double cy,
    double u_obs, double v_obs,
    Eigen::Matrix<double,2,6>* J_cam = nullptr,
    Eigen::Matrix<double,2,3>* J_pt  = nullptr)
{
    bool need_jac = (J_cam != nullptr) || (J_pt != nullptr);
    Eigen::Matrix3d dR[3];
    Eigen::Matrix3d R = rodrigues(rvec, need_jac ? dR : nullptr);

    Eigen::Vector3d Xc = R * X + t;
    double xc=Xc[0], yc=Xc[1], zc=Xc[2];
    double inv_z  = 1.0 / zc;
    double inv_z2 = inv_z * inv_z;

    Eigen::Vector2d res(fx*xc*inv_z + cx - u_obs,
                        fy*yc*inv_z + cy - v_obs);

    if (!need_jac) return res;

    // d([u,v])/d([xc,yc,zc])
    Eigen::Matrix<double,2,3> dproj;
    dproj << fx*inv_z,       0,  -fx*xc*inv_z2,
                    0,  fy*inv_z,  -fy*yc*inv_z2;

    if (J_pt)  *J_pt  = dproj * R;       // d(Xc)/dX = R

    if (J_cam) {
        J_cam->block<2,3>(0,3) = dproj;  // d(Xc)/dt = I
        for (int i = 0; i < 3; ++i) {
            // d(Xc)/dr_i = dR[i] * X
            J_cam->col(i) = dproj * (dR[i] * X);
        }
    }
    return res;
}

// ── Huber weight (robust kernel) ──────────────────────────────────────────────
static inline double huber_weight(double r_norm, double delta = 1.0) {
    return r_norm <= delta ? 1.0 : delta / r_norm;
}

// ── Schur-complement Levenberg-Marquardt BA ───────────────────────────────────
namespace ba {

BAResult run_ba(
    const std::vector<double>& cam_params_in,
    const std::vector<double>& pts3d_in,
    const std::vector<Observation>& obs,
    double fx, double fy, double cx, double cy,
    int max_iter, double lm_lambda0, double tol)
{
    const int NC = (int)(cam_params_in.size() / 6);
    const int NP = (int)(pts3d_in.size()  / 3);
    const int M  = (int)obs.size();

    std::vector<double> cams = cam_params_in;
    std::vector<double> pts  = pts3d_in;

    // Group observations by point (for back-substitution)
    std::vector<std::vector<int>> pt_obs(NP);
    for (int k = 0; k < M; ++k)
        pt_obs[obs[k].pt_idx].push_back(k);

    // Per-point Hessian and gradient storage
    std::vector<Eigen::Matrix<double,6,6>> Hcc(NC);
    std::vector<Eigen::Matrix<double,3,3>> Hpp(NP);
    std::vector<Eigen::Matrix<double,6,3>> Hcp(M);
    std::vector<Eigen::Matrix<double,6,1>> bc(NC);
    std::vector<Eigen::Matrix<double,3,1>> bp(NP);

    double lambda = lm_lambda0;
    double last_cost = std::numeric_limits<double>::max();
    int n_iters = 0;

    auto cam_rvec = [&](const std::vector<double>& c, int ci) -> Eigen::Vector3d {
        return Eigen::Vector3d(c.data() + ci*6);
    };
    auto cam_t = [&](const std::vector<double>& c, int ci) -> Eigen::Vector3d {
        return Eigen::Vector3d(c.data() + ci*6 + 3);
    };
    auto pt_X = [&](const std::vector<double>& p, int pi) -> Eigen::Vector3d {
        return Eigen::Vector3d(p.data() + pi*3);
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        // ── Build H and b ─────────────────────────────────────────────────
        for (int ci = 0; ci < NC; ++ci) { Hcc[ci].setZero(); bc[ci].setZero(); }
        for (int pi = 0; pi < NP; ++pi) { Hpp[pi].setZero(); bp[pi].setZero(); }

        double cost = 0.0;
        for (int k = 0; k < M; ++k) {
            int ci = obs[k].cam_idx, pi = obs[k].pt_idx;
            Eigen::Matrix<double,2,6> Jc;
            Eigen::Matrix<double,2,3> Jp;
            Eigen::Vector2d r = project_residual(
                cam_rvec(cams,ci), cam_t(cams,ci), pt_X(pts,pi),
                fx, fy, cx, cy, obs[k].u, obs[k].v, &Jc, &Jp);

            double w  = huber_weight(r.norm());
            double w2 = w * w;
            cost += 0.5 * w2 * r.squaredNorm();

            Hcc[ci].noalias() += w2 * Jc.transpose() * Jc;
            Hpp[pi].noalias() += w2 * Jp.transpose() * Jp;
            Hcp[k]             = w2 * Jc.transpose() * Jp;
            bc[ci].noalias()  += w2 * (Jc.transpose() * r);
            bp[pi].noalias()  += w2 * (Jp.transpose() * r);
        }

        // ── Precompute (Hpp + λI)^{-1} per point ────────────────────────
        std::vector<Eigen::Matrix3d> Hpp_inv(NP);
        for (int pi = 0; pi < NP; ++pi) {
            Eigen::Matrix3d H = Hpp[pi];
            H(0,0)+=lambda; H(1,1)+=lambda; H(2,2)+=lambda;
            Hpp_inv[pi] = H.inverse();
        }

        // ── Build Schur complement S and b_s ────────────────────────────
        // S   = (Hcc+λI) - Hcp * (Hpp+λI)^{-1} * Hcp^T
        // b_s = bc      - Hcp * (Hpp+λI)^{-1} * bp
        int Ndim = NC * 6;
        Eigen::MatrixXd S  = Eigen::MatrixXd::Zero(Ndim, Ndim);
        Eigen::VectorXd bs = Eigen::VectorXd::Zero(Ndim);

        for (int ci = 0; ci < NC; ++ci) {
            S.block<6,6>(ci*6, ci*6) = Hcc[ci];
            S.block<6,6>(ci*6, ci*6).diagonal().array() += lambda;
            bs.segment<6>(ci*6) = bc[ci];
        }

        for (int pi = 0; pi < NP; ++pi) {
            const auto& kidxs = pt_obs[pi];
            if (kidxs.empty()) continue;
            const Eigen::Matrix3d& Hinv = Hpp_inv[pi];

            for (int ki : kidxs) {
                int ci = obs[ki].cam_idx;
                Eigen::Matrix<double,6,3> EH = Hcp[ki] * Hinv;  // 6×3

                // b_s[ci] -= Hcp[ki] * Hpp_inv * bp[pi]
                bs.segment<6>(ci*6).noalias() -= EH * bp[pi];

                // S[ci, cj] -= Hcp[ki] * Hpp_inv * Hcp[kj]^T  for all kj sharing pt pi
                for (int kj : kidxs) {
                    int cj = obs[kj].cam_idx;
                    S.block<6,6>(ci*6, cj*6).noalias() -= EH * Hcp[kj].transpose();
                }
            }
        }

        // ── Solve for camera update ────────────────────────────────────
        Eigen::VectorXd delta_c = S.ldlt().solve(-bs);

        // ── Back-substitute: delta_p = -(Hpp+λI)^{-1} * (bp + Hcp^T * delta_c)
        std::vector<Eigen::Vector3d> delta_p(NP, Eigen::Vector3d::Zero());
        for (int pi = 0; pi < NP; ++pi) {
            const auto& kidxs = pt_obs[pi];
            if (kidxs.empty()) continue;
            Eigen::Vector3d rhs = bp[pi];
            for (int ki : kidxs) {
                int ci = obs[ki].cam_idx;
                rhs.noalias() += Hcp[ki].transpose() * delta_c.segment<6>(ci*6);
            }
            delta_p[pi] = -Hpp_inv[pi] * rhs;
        }

        // ── Trial update ──────────────────────────────────────────────
        std::vector<double> cams_new = cams, pts_new = pts;
        for (int ci = 0; ci < NC; ++ci)
            for (int d = 0; d < 6; ++d)
                cams_new[ci*6+d] += delta_c[ci*6+d];
        for (int pi = 0; pi < NP; ++pi)
            for (int d = 0; d < 3; ++d)
                pts_new[pi*3+d] += delta_p[pi][d];

        // Evaluate new cost
        double new_cost = 0.0;
        for (int k = 0; k < M; ++k) {
            int ci = obs[k].cam_idx, pi = obs[k].pt_idx;
            Eigen::Vector2d r = project_residual(
                cam_rvec(cams_new,ci), cam_t(cams_new,ci), pt_X(pts_new,pi),
                fx, fy, cx, cy, obs[k].u, obs[k].v);
            double w = huber_weight(r.norm());
            new_cost += 0.5 * w*w * r.squaredNorm();
        }

        if (new_cost < cost) {
            cams = cams_new;
            pts  = pts_new;
            lambda = std::max(lambda * 0.1, 1e-10);
            last_cost = new_cost;
            ++n_iters;
            if (delta_c.norm() < tol) break;
        } else {
            lambda = std::min(lambda * 10.0, 1e10);
        }
        (void)last_cost;
    }

    // Final mean reprojection error
    double sum_sq = 0.0;
    for (int k = 0; k < M; ++k) {
        int ci = obs[k].cam_idx, pi = obs[k].pt_idx;
        Eigen::Vector2d r = project_residual(
            cam_rvec(cams,ci), cam_t(cams,ci), pt_X(pts,pi),
            fx, fy, cx, cy, obs[k].u, obs[k].v);
        sum_sq += r.squaredNorm();
    }

    BAResult result;
    result.cam_params      = cams;
    result.pts3d           = pts;
    result.mean_reproj_err = std::sqrt(sum_sq / M);
    result.n_iters         = n_iters;
    return result;
}

} // namespace ba
