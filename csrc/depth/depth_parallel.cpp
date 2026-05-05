#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <vector>
#include <atomic>
#include <stdexcept>

namespace py = pybind11;

// ── Pure-C++ remap grid builder ───────────────────────────────────────────────
// Replaces the Python ray-tracing loop in compute_depth_map:
//   rays_rect = R1 @ rays.reshape(-1,3).T  →  then project with P1
// Returns (map_x, map_y) as float32 numpy arrays ready for cv2.remap.
py::tuple build_remap_grid(
    int H, int W,
    py::array_t<double> K_arr,     // 3×3
    py::array_t<double> R1_arr,    // 3×3  rectification rotation
    py::array_t<double> P1_arr)    // 3×4  rectified projection
{
    auto K  = K_arr.unchecked<2>();
    auto R1 = R1_arr.unchecked<2>();
    auto P1 = P1_arr.unchecked<2>();

    double fx_orig = K(0,0), fy_orig = K(1,1);
    double cx_orig = K(0,2), cy_orig = K(1,2);
    double fx_rect = P1(0,0), fy_rect = P1(1,1);
    double cx_rect = P1(0,2), cy_rect = P1(1,2);

    // R1 as Eigen matrix
    Eigen::Matrix3d R1e;
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) R1e(r,c) = R1(r,c);

    py::array_t<float> map_x({H, W});
    py::array_t<float> map_y({H, W});
    auto mx = map_x.mutable_unchecked<2>();
    auto my = map_y.mutable_unchecked<2>();

    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            // Normalized ray in original camera
            Eigen::Vector3d ray(
                (u - cx_orig) / fx_orig,
                (v - cy_orig) / fy_orig,
                1.0);
            // Apply rectification rotation
            Eigen::Vector3d rr = R1e * ray;
            double z = rr[2];
            if (std::abs(z) < 1e-6) z = 1e-6;
            mx(v, u) = float(rr[0] / z * fx_rect + cx_rect);
            my(v, u) = float(rr[1] / z * fy_rect + cy_rect);
        }
    }
    return py::make_tuple(map_x, map_y);
}

// ── Simple threadpool ─────────────────────────────────────────────────────────
struct ThreadPool {
    std::vector<std::thread>          workers;
    std::queue<std::function<void()>> tasks;
    std::mutex                        mtx;
    std::condition_variable           cv;
    bool                              stop = false;

    explicit ThreadPool(int n) {
        for (int i = 0; i < n; ++i)
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv.wait(lk, [this]{ return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
    }

    void enqueue(std::function<void()> f) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            tasks.push(std::move(f));
        }
        cv.notify_one();
    }

    ~ThreadPool() {
        { std::unique_lock<std::mutex> lk(mtx); stop = true; }
        cv.notify_all();
        for (auto& w : workers) w.join();
    }
};

// ── Parallel estimate_all_depths ──────────────────────────────────────────────
// compute_depth_fn: Python callable with signature:
//   compute_depth_fn(img_i, img_j, K, R_i, t_i, R_j, t_j) -> depth_map
// Returns list of fused depth maps, one per image.
py::list estimate_all_depths_parallel(
    py::list images,           // list of numpy arrays (HxWx3 uint8)
    py::list poses,            // list of dicts {"K":..., "R":..., "t":...}
    py::object compute_depth_fn,
    int neighbors   = 2,
    int n_threads   = 0)       // 0 = auto (cpu_count)
{
    int n = (int)images.size();
    if (n_threads <= 0)
        n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, n);

    // Per-image depth results: vector of vector<py::object> (one per valid pair)
    std::vector<std::vector<py::object>> depth_lists(n);
    std::mutex results_mutex;  // guards depth_lists
    std::vector<std::string> errors(n);
    std::atomic<int> n_errors{0};

    // Build all (i, j) pairs upfront
    struct Pair { int i, j; };
    std::vector<Pair> pairs;
    for (int i = 0; i < n; ++i)
        for (int delta = 1; delta <= neighbors; ++delta)
            for (int j : {i - delta, i + delta})
                if (j >= 0 && j < n)
                    pairs.push_back({i, j});

    {
        // Release GIL before spawning threads so worker threads can acquire it.
        py::gil_scoped_release release_main;

        ThreadPool pool(n_threads);

        for (const auto& p : pairs) {
            pool.enqueue([&, p] {
                // Use raw CPython GIL API — safer for non-Python threads
                PyGILState_STATE gstate = PyGILState_Ensure();
                try {
                    py::object img_i   = images[p.i];
                    py::object img_j   = images[p.j];
                    py::object pose_i  = poses[p.i];
                    py::object pose_j  = poses[p.j];
                    py::object np      = py::module_::import("numpy");
                    py::object K_      = np.attr("array")(pose_i["K"]);
                    py::object R_i     = np.attr("array")(pose_i["R"]);
                    py::object t_i     = np.attr("array")(pose_i["t"]);
                    py::object R_j     = np.attr("array")(pose_j["R"]);
                    py::object t_j     = np.attr("array")(pose_j["t"]);

                    // Calls into Python/OpenCV; OpenCV releases GIL internally during SGBM
                    py::object depth = compute_depth_fn(img_i, img_j, K_, R_i, t_i, R_j, t_j);

                    py::object mask  = np.attr("greater")(depth, py::int_(0));
                    int valid_count  = np.attr("sum")(mask).cast<int>();
                    if (valid_count > 200) {
                        std::lock_guard<std::mutex> lk(results_mutex);
                        depth_lists[p.i].push_back(std::move(depth));
                    }
                } catch (...) {
                    // Skip bad pairs (degenerate baseline, memory, etc.)
                }
                PyGILState_Release(gstate);
            });
        }
    }  // ThreadPool destructor joins all threads; release_main goes out of scope → GIL re-acquired

    // Fuse per-image depth lists (GIL held again here)
    py::list result;
    {
        py::object np = py::module_::import("numpy");

        for (int i = 0; i < n; ++i) {
            auto& depths = depth_lists[i];
            py::object fused;

            if (depths.empty()) {
                auto img  = py::array(images[i]);
                auto shape = img.attr("shape").cast<py::tuple>();
                int H = shape[0].cast<int>(), W = shape[1].cast<int>();
                fused = np.attr("zeros")(
                    py::make_tuple(H, W),
                    py::arg("dtype") = np.attr("float32"));
            } else {
                py::list dlist;
                for (auto& d : depths) dlist.append(d);
                py::object stack  = np.attr("stack")(dlist, py::int_(0));
                py::object valid  = np.attr("greater")(stack, py::int_(0));
                py::object count  = np.attr("sum")(valid, py::int_(0)).attr("astype")(np.attr("float32"));
                py::object count_safe = np.attr("where")(
                    np.attr("equal")(count, py::float_(0.0)),
                    py::float_(1.0), count);
                fused = np.attr("sum")(np.attr("multiply")(stack, valid),
                                       py::int_(0)) / count_safe;
            }

            int valid_px = np.attr("sum")(
                np.attr("greater")(fused, py::int_(0))).cast<int>();
            py::print(py::str("[depth/par] img {:04d}: valid pixels = {}").format(
                py::int_(i), py::int_(valid_px)));

            result.append(fused);
        }
    }
    return result;
}

PYBIND11_MODULE(_depth_cpp, m) {
    m.doc() = "C++ parallel MVS depth estimation helpers";

    m.def("build_remap_grid", &build_remap_grid,
          py::arg("H"), py::arg("W"),
          py::arg("K"), py::arg("R1"), py::arg("P1"),
          "Build (map_x, map_y) float32 arrays for cv2.remap back-projection.");

    m.def("estimate_all_depths_parallel", &estimate_all_depths_parallel,
          py::arg("images"),
          py::arg("poses"),
          py::arg("compute_depth_fn"),
          py::arg("neighbors") = 2,
          py::arg("n_threads") = 0,
          "Parallel depth estimation: dispatches stereo pairs across C++ threads.");
}
