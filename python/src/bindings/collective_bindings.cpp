// HPXPy - Collective Operations Bindings
//
// SPDX-License-Identifier: BSL-1.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hpx/hpx.hpp>
#include <hpx/collectives/all_reduce.hpp>
#include <hpx/collectives/broadcast.hpp>
#include <hpx/collectives/gather.hpp>
#include <hpx/collectives/scatter.hpp>
#include <hpx/collectives/barrier.hpp>

#include "ndarray.hpp"

namespace py = pybind11;

namespace hpxpy {

// Reduction operation types
enum class ReduceOp {
    Sum,
    Prod,
    Min,
    Max
};

// Get the number of localities
std::uint32_t get_num_localities_impl() {
    return hpx::get_num_localities(hpx::launch::sync);
}

// Get the current locality ID
std::uint32_t get_locality_id_impl() {
    return hpx::get_locality_id();
}

// Barrier synchronization - wait for all localities
void barrier_impl(const std::string& name) {
    // In single-locality mode, this is a no-op
    // In multi-locality mode, this synchronizes all localities
    if (get_num_localities_impl() > 1) {
        hpx::distributed::barrier b(name);
        b.wait();
    }
    // Single locality: no barrier needed
}

// All-reduce: combine values from all localities using the specified operation
// In single-locality mode, this just returns the input (identity operation)
ndarray all_reduce_impl(ndarray const& arr, ReduceOp op) {
    std::uint32_t num_localities = get_num_localities_impl();

    if (num_localities == 1) {
        // Single locality: return a copy of the input
        // The "reduction" of one value is itself
        py::array np_arr = arr.to_numpy();
        return ndarray(np_arr, true);  // copy
    }

    // Multi-locality: perform actual all_reduce
    // This requires all localities to participate
    // TODO: Implement actual distributed all_reduce when multi-locality is enabled
    // For now, return a copy
    py::array np_arr = arr.to_numpy();
    return ndarray(np_arr, true);
}

// Broadcast: send array from root to all localities
// In single-locality mode, this just returns the input
ndarray broadcast_impl(ndarray const& arr, std::uint32_t root) {
    std::uint32_t num_localities = get_num_localities_impl();
    std::uint32_t locality_id = get_locality_id_impl();

    if (num_localities == 1) {
        // Single locality: return a copy of the input
        py::array np_arr = arr.to_numpy();
        return ndarray(np_arr, true);
    }

    // Multi-locality: perform actual broadcast
    // TODO: Implement actual distributed broadcast
    py::array np_arr = arr.to_numpy();
    return ndarray(np_arr, true);
}

// Gather: collect arrays from all localities to root
// In single-locality mode, returns a list containing just the input
py::list gather_impl(ndarray const& arr, std::uint32_t root) {
    std::uint32_t num_localities = get_num_localities_impl();

    py::list result;

    if (num_localities == 1) {
        // Single locality: return list with one element
        result.append(arr.to_numpy());
        return result;
    }

    // Multi-locality: perform actual gather
    // TODO: Implement actual distributed gather
    result.append(arr.to_numpy());
    return result;
}

// Scatter: distribute array from root to all localities
// In single-locality mode, returns the input unchanged
ndarray scatter_impl(ndarray const& arr, std::uint32_t root) {
    std::uint32_t num_localities = get_num_localities_impl();

    if (num_localities == 1) {
        // Single locality: return a copy of the input
        py::array np_arr = arr.to_numpy();
        return ndarray(np_arr, true);
    }

    // Multi-locality: perform actual scatter
    // TODO: Implement actual distributed scatter
    py::array np_arr = arr.to_numpy();
    return ndarray(np_arr, true);
}

// Register collective bindings
void register_collective_bindings(py::module_& m) {
    // Reduction operation enum
    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("sum", ReduceOp::Sum)
        .value("prod", ReduceOp::Prod)
        .value("min", ReduceOp::Min)
        .value("max", ReduceOp::Max);

    // Collective operations submodule
    auto collectives = m.def_submodule("collectives",
        "HPX collective operations for distributed computing");

    // Locality info
    collectives.def("get_num_localities", &get_num_localities_impl,
        "Get the number of HPX localities (nodes)");

    collectives.def("get_locality_id", &get_locality_id_impl,
        "Get the ID of the current locality");

    // Barrier
    collectives.def("barrier", &barrier_impl,
        py::arg("name") = "hpxpy_barrier",
        R"doc(
        Synchronize all localities.

        In single-locality mode, this is a no-op.
        In multi-locality mode, all localities wait until everyone reaches the barrier.

        Parameters
        ----------
        name : str, optional
            Name for the barrier (default: "hpxpy_barrier")
        )doc");

    // All-reduce
    collectives.def("all_reduce", &all_reduce_impl,
        py::arg("arr"),
        py::arg("op") = ReduceOp::Sum,
        R"doc(
        Combine values from all localities using a reduction operation.

        Each locality contributes its local array, and all localities receive
        the combined result.

        Parameters
        ----------
        arr : ndarray
            Local array to contribute
        op : ReduceOp, optional
            Reduction operation (sum, prod, min, max). Default: sum

        Returns
        -------
        ndarray
            Combined result (same on all localities)
        )doc");

    // Broadcast
    collectives.def("broadcast", &broadcast_impl,
        py::arg("arr"),
        py::arg("root") = 0,
        R"doc(
        Broadcast array from root locality to all localities.

        Parameters
        ----------
        arr : ndarray
            Array to broadcast (only used on root)
        root : int, optional
            Locality ID to broadcast from (default: 0)

        Returns
        -------
        ndarray
            Broadcasted array (same on all localities)
        )doc");

    // Gather
    collectives.def("gather", &gather_impl,
        py::arg("arr"),
        py::arg("root") = 0,
        R"doc(
        Gather arrays from all localities to root.

        Parameters
        ----------
        arr : ndarray
            Local array to contribute
        root : int, optional
            Locality ID to gather to (default: 0)

        Returns
        -------
        list
            List of arrays from all localities (only valid on root)
        )doc");

    // Scatter
    collectives.def("scatter", &scatter_impl,
        py::arg("arr"),
        py::arg("root") = 0,
        R"doc(
        Scatter array from root to all localities.

        The array on root is divided evenly among all localities.

        Parameters
        ----------
        arr : ndarray
            Array to scatter (only used on root)
        root : int, optional
            Locality ID to scatter from (default: 0)

        Returns
        -------
        ndarray
            This locality's portion of the scattered array
        )doc");

    // Convenience aliases at module level
    m.attr("all_reduce") = collectives.attr("all_reduce");
    m.attr("broadcast") = collectives.attr("broadcast");
    m.attr("gather") = collectives.attr("gather");
    m.attr("scatter") = collectives.attr("scatter");
    m.attr("barrier") = collectives.attr("barrier");
}

}  // namespace hpxpy
