//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

inline double sqr(double val)
{
    return val * val;
}

int main(int argc, char* argv[])
{
    std::size_t N = 1'000'000;
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t locality_id = hpx::get_locality_id();

    if (locality_id == 0 && argc > 1)
        N = std::stol(argv[1]);

    hpx::collectives::broadcast(hpx::collectives::get_world_communicator(), N);

    std::size_t const blocksize = N / num_localities;
    std::size_t const begin = blocksize * locality_id;
    std::size_t const end = blocksize * (locality_id + 1);
    double h = 1.0 / static_cast<double>(N);

    double pi = 0.0;
    for (std::size_t i = begin; i != end; ++i)
        pi += h * 4.0 / (1 + sqr(static_cast<double>(i) * h));

    hpx::collectives::reduce(
        hpx::collectives::get_world_communicator(), pi, std::plus{});

    if (locality_id == 0)
        std::cout << "pi: " << pi << std::endl;

    return 0;
}

#else

int main(int argc, char* argv[])
{
    return 0;
}

#endif
