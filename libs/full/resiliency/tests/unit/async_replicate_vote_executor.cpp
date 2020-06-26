//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2019 Nikunj Gupta
//  Copyright (c) 2018-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <random>
#include <vector>

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(1.0, 10.0);

int vote(std::vector<int> vect)
{
    return vect.at(0);
}

int universal_ans()
{
    if (dist(mt) > 5)
        return 42;
    return 84;
}

bool validate(int ans)
{
    return ans == 42;
}

int hpx_main()
{
    {
        hpx::parallel::execution::parallel_executor exec;

        hpx::future<int> f =
            hpx::resiliency::experimental::async_replicate_vote(
                exec, 10, &vote, &universal_ans);

        auto result = f.get();
        HPX_TEST(result == 42 || result == 84);
    }

    {
        hpx::parallel::execution::parallel_executor exec;

        hpx::future<int> f =
            hpx::resiliency::experimental::async_replicate_vote_validate(
                exec, 10, &vote, &validate, &universal_ans);

        auto result = f.get();
        HPX_TEST(result == 42 || result == 84);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST(hpx::init(argc, argv) == 0);
    return hpx::util::report_errors();
}
