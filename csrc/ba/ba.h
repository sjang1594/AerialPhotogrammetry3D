#pragma once
#include <Eigen/Core>
#include <vector>
#include <cstdint>

namespace ba {

struct Observation {
    int32_t cam_idx;
    int32_t pt_idx;
    double  u, v;       // observed 2-D pixel
};

// Result returned to Python
struct BAResult {
    std::vector<double> cam_params;  // n_cams * 6  [rvec(3), t(3)]
    std::vector<double> pts3d;       // n_pts  * 3
    double mean_reproj_err;
    int    n_iters;
};

BAResult run_ba(
    const std::vector<double>& cam_params_in,   // n_cams * 6
    const std::vector<double>& pts3d_in,        // n_pts  * 3
    const std::vector<Observation>& obs,
    double fx, double fy, double cx, double cy,
    int    max_iter   = 50,
    double lm_lambda0 = 1e-3,
    double tol        = 1e-6
);

} // namespace ba
