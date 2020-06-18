// Copyright (c) 2020 Nikunj Gupta
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/compute.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/util.hpp>


#include <algorithm>
#include <array>

using allocator_type = hpx::compute::host::block_allocator<double>;
using data_type = hpx::compute::vector<double, allocator_type>;

const double k = 0.5;     // heat transfer coefficient
const double dt = 1.;     // time step
const double dx = 1.;     // grid spacing

template <typename Container>
void init(
    std::array<Container, 2>& U, std::size_t Nx,
    std::size_t rank = 0, std::size_t num_localities = 1)
{
    // Initialize: Boundaries are set to 1, interior is 0
    if (rank == 0)
    {
        U[0][0] = 1.0;
        U[1][0] = 1.0;
    }
    if (rank == num_localities - 1)
    {
        U[0][Nx-1] = 100.0;
        U[1][Nx-1] = 100.0;
    }
}

void stencil_update(std::array<data_type, 2>& U, const std::size_t& begin
        , const std::size_t& end, const std::size_t t)
{
    data_type& curr = U[t % 2];
    data_type& next = U[(t + 1) % 2];

    #pragma unroll
    for (std::size_t i = begin; i < end; ++i)
    {
        // next[i] = (curr[i-1] + curr[i+1]) * 0.5f;
        next[i] = curr[i] + ((k*dt)/(dx*dx)) * (curr[i-1] - 2*curr[i] + curr[i+1]);
    }
}