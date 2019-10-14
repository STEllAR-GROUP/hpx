//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_executors.hpp>

#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct my_executor : hpx::parallel::execution::parallel_executor
{
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_one_way_executor<my_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<my_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<my_executor> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;

    my_executor exec;

    std::vector<int> v(100);

    for_each(par.on(exec), v.begin(), v.end(), [](int x) {});

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
