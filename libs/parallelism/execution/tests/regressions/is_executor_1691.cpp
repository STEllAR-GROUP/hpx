//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct my_executor : hpx::execution::parallel_executor
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
int hpx_main()
{
    my_executor exec;

    std::vector<int> v(100);

    hpx::ranges::for_each(hpx::execution::par.on(exec), v, [](int) {});

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
