//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test for GitHub issue #6740: a CUDA translation unit must be able
// to name HPX types and functions from the collectives, colocated and
// component-factory APIs in host member functions of a class that also owns a
// __global__ kernel. Compile-only smoke test; a successful build is the pass.

#include <hpx/async_colocated/post_colocated.hpp>
#include <hpx/collectives.hpp>
#include <hpx/runtime_configuration/component_factory_base.hpp>

__global__ void k() {}

// 1. Collectives shape — spans the headers whose whole-file guards were
//    removed: create_communicator, barrier, broadcast, all_reduce, all_gather,
//    scatter.
struct collectives_shape
{
    void host_side()
    {
        hpx::collectives::communicator c;
        (void) c;

        using hpx::collectives::all_gather;
        using hpx::collectives::all_reduce;
        using hpx::collectives::barrier;
        using hpx::collectives::broadcast_to;
        using hpx::collectives::scatter_to;
    }
    void launch()
    {
        k<<<1, 1>>>();
    }
};

// 2. post_colocated shape — the whole-file guard in post_colocated.hpp used
//    to hide the definitions of hpx::detail::post_colocated. Compile-only,
//    since a real call would only surface as a link-time reference here.
struct post_colocated_shape
{
    void host_side()
    {
        using hpx::detail::post_colocated;
    }
    void launch()
    {
        k<<<1, 1>>>();
    }
};

// 3. component_factory_base shape — small guard used to hide the forward
//    declaration of the struct on the device pass.
struct component_factory_base_shape
{
    void host_side()
    {
        hpx::components::component_factory_base* p = nullptr;
        (void) p;
    }
    void launch()
    {
        k<<<1, 1>>>();
    }
};

int main()
{
    return 0;
}
