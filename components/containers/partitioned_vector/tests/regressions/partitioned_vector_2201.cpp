//  Copyright (c) 2016 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <cstddef>
#include <random>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(int);
unsigned int seed = std::random_device{}();

///////////////////////////////////////////////////////////////////////////////
struct random_fill
{
    random_fill()
      : gen(seed),
        dist(0, RAND_MAX)
    {}

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {}
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (hpx::get_locality_id() == 0)
    {
        // create as many partitions as we have localities
        std::size_t size = 10000;
        hpx::partitioned_vector<int> v(
            size, hpx::container_layout(hpx::find_all_localities()));

        // initialize data
        // segmented version of algorithm used
        using namespace  hpx::parallel;
        generate(execution::par, v.begin(), v.end(), random_fill());

        return hpx::finalize();
    }

    return 0;
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
