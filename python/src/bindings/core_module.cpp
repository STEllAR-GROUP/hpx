// HPXPy - High Performance Python Arrays powered by HPX
//
// SPDX-License-Identifier: BSL-1.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations for binding functions
void bind_runtime(py::module_& m);
void bind_array(py::module_& m);
void bind_algorithms(py::module_& m);
void bind_execution(py::module_& m);
void bind_gpu(py::module_& m);

namespace hpxpy {
void register_collective_bindings(py::module_& m);
void register_distributed_array_bindings(py::module_& m);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        HPXPy Core Extension Module
        ---------------------------

        This module provides the C++ bindings for HPX functionality,
        including runtime management, array operations, and parallel algorithms.
    )pbdoc";

    // Version information
    m.attr("__version__") = HPXPY_VERSION;
    m.attr("__version_info__") = py::make_tuple(
        HPXPY_VERSION_MAJOR,
        HPXPY_VERSION_MINOR,
        HPXPY_VERSION_PATCH
    );

    // Bind all submodules
    bind_runtime(m);
    bind_array(m);
    bind_algorithms(m);
    bind_execution(m);
    hpxpy::register_collective_bindings(m);
    hpxpy::register_distributed_array_bindings(m);
    bind_gpu(m);

    // GPU availability flags
#ifdef HPXPY_HAVE_CUDA
    m.attr("_has_cuda") = true;
#else
    m.attr("_has_cuda") = false;
#endif

#ifdef HPXPY_HAVE_SYCL
    m.attr("_has_sycl") = true;
#else
    m.attr("_has_sycl") = false;
#endif
}
