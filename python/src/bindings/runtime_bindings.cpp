// HPXPy - Runtime bindings
//
// SPDX-License-Identifier: BSL-1.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hpx/hpx_init.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/runtime.hpp>
#include <hpx/modules/runtime_local.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace py = pybind11;

namespace hpxpy {

// Global state for runtime management
namespace {
    std::atomic<bool> g_initialized{false};
    std::atomic<bool> g_running{false};
    std::mutex g_init_mutex;
}

// Initialize the HPX runtime
void init(py::object num_threads_obj, std::vector<std::string> const& config) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (g_initialized.load()) {
        throw std::runtime_error("HPX runtime is already initialized");
    }

    // Build command line arguments
    std::vector<std::string> args;
    args.push_back("hpxpy");  // Program name

    // Handle num_threads
    if (!num_threads_obj.is_none()) {
        int num_threads = num_threads_obj.cast<int>();
        args.push_back("--hpx:threads=" + std::to_string(num_threads));
    }

    // Add user-provided configuration
    for (auto const& cfg : config) {
        args.push_back(cfg);
    }

    // Convert to argc/argv format
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    int argc = static_cast<int>(argv.size() - 1);
    char** argv_ptr = argv.data();

    // Release GIL during HPX initialization (can take time)
    py::gil_scoped_release release;

    // Initialize HPX runtime
    // Using hpx::start for non-blocking initialization
    hpx::start(nullptr, argc, argv_ptr);

    g_initialized.store(true);
    g_running.store(true);
}

// Finalize the HPX runtime
void finalize() {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (!g_initialized.load()) {
        throw std::runtime_error("HPX runtime is not initialized");
    }

    if (!g_running.load()) {
        return;  // Already finalized
    }

    // Release GIL during finalization
    py::gil_scoped_release release;

    // Stop the HPX runtime
    hpx::finalize();
    hpx::stop();

    g_running.store(false);
    g_initialized.store(false);
}

// Check if runtime is currently running
bool is_running() {
    return g_running.load();
}

// Get number of OS threads
int num_threads() {
    if (!g_running.load()) {
        throw std::runtime_error("HPX runtime is not running");
    }
    return hpx::get_num_worker_threads();
}

// Get number of localities
int num_localities() {
    if (!g_running.load()) {
        throw std::runtime_error("HPX runtime is not running");
    }
    return static_cast<int>(hpx::get_num_localities(hpx::launch::sync));
}

// Get current locality ID
int locality_id() {
    if (!g_running.load()) {
        throw std::runtime_error("HPX runtime is not running");
    }
    return static_cast<int>(hpx::get_locality_id());
}

}  // namespace hpxpy

void bind_runtime(py::module_& m) {
    m.def("init", &hpxpy::init,
          py::arg("num_threads") = py::none(),
          py::arg("config") = std::vector<std::string>{},
          R"pbdoc(
              Initialize the HPX runtime system.

              Parameters
              ----------
              num_threads : int, optional
                  Number of OS threads to use.
              config : list of str, optional
                  Additional HPX configuration options.
          )pbdoc");

    m.def("finalize", &hpxpy::finalize,
          R"pbdoc(
              Finalize the HPX runtime system.
          )pbdoc");

    m.def("is_running", &hpxpy::is_running,
          R"pbdoc(
              Check if the HPX runtime is currently running.

              Returns
              -------
              bool
                  True if runtime is initialized and running.
          )pbdoc");

    m.def("num_threads", &hpxpy::num_threads,
          R"pbdoc(
              Get the number of OS threads used by HPX.

              Returns
              -------
              int
                  Number of worker threads.
          )pbdoc");

    m.def("num_localities", &hpxpy::num_localities,
          R"pbdoc(
              Get the number of localities (nodes) in the HPX runtime.

              Returns
              -------
              int
                  Number of localities.
          )pbdoc");

    m.def("locality_id", &hpxpy::locality_id,
          R"pbdoc(
              Get the ID of the current locality.

              Returns
              -------
              int
                  Current locality ID (0-based).
          )pbdoc");
}
