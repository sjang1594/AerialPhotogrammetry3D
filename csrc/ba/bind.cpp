#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ba.h"

namespace py = pybind11;

// Convert SfMReconstruction to flat arrays, call C++ BA, write back.
py::dict run_bundle_adjustment_cpp(
    py::array_t<double> cam_params,   // (NC, 6)  [rvec(3) | t(3)]
    py::array_t<double> pts3d,        // (NP, 3)
    py::array_t<int32_t> obs_cam,     // (M,)
    py::array_t<int32_t> obs_pt,      // (M,)
    py::array_t<double>  obs_xy,      // (M, 2)
    double fx, double fy, double cx, double cy,
    int max_iter)
{
    auto rc = cam_params.unchecked<2>();
    auto rp = pts3d.unchecked<2>();
    auto ro_cam = obs_cam.unchecked<1>();
    auto ro_pt  = obs_pt.unchecked<1>();
    auto ro_xy  = obs_xy.unchecked<2>();

    int NC = (int)rc.shape(0);
    int NP = (int)rp.shape(0);
    int M  = (int)ro_cam.shape(0);

    std::vector<double> cp(NC * 6), pt(NP * 3);
    for (int i = 0; i < NC; ++i)
        for (int d = 0; d < 6; ++d) cp[i*6+d] = rc(i,d);
    for (int i = 0; i < NP; ++i)
        for (int d = 0; d < 3; ++d) pt[i*3+d] = rp(i,d);

    std::vector<ba::Observation> obs(M);
    for (int k = 0; k < M; ++k) {
        obs[k].cam_idx = ro_cam(k);
        obs[k].pt_idx  = ro_pt(k);
        obs[k].u = ro_xy(k, 0);
        obs[k].v = ro_xy(k, 1);
    }

    ba::BAResult result = ba::run_ba(cp, pt, obs, fx, fy, cx, cy, max_iter);

    // Return updated arrays
    py::array_t<double> out_cams({NC, 6});
    py::array_t<double> out_pts({NP, 3});
    auto wc = out_cams.mutable_unchecked<2>();
    auto wp = out_pts.mutable_unchecked<2>();
    for (int i = 0; i < NC; ++i)
        for (int d = 0; d < 6; ++d) wc(i,d) = result.cam_params[i*6+d];
    for (int i = 0; i < NP; ++i)
        for (int d = 0; d < 3; ++d) wp(i,d) = result.pts3d[i*3+d];

    py::dict out;
    out["cam_params"]       = out_cams;
    out["pts3d"]            = out_pts;
    out["mean_reproj_err"]  = result.mean_reproj_err;
    out["n_iters"]          = result.n_iters;
    return out;
}

PYBIND11_MODULE(_ba_cpp, m) {
    m.doc() = "C++ Schur-LM Bundle Adjustment";
    m.def("run_bundle_adjustment_cpp", &run_bundle_adjustment_cpp,
          py::arg("cam_params"),
          py::arg("pts3d"),
          py::arg("obs_cam"),
          py::arg("obs_pt"),
          py::arg("obs_xy"),
          py::arg("fx"), py::arg("fy"),
          py::arg("cx"), py::arg("cy"),
          py::arg("max_iter") = 50);
}
