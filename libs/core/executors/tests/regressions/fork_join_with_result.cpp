//  Copyright (c) 2023 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/numeric.hpp>

#include <functional>

int hpx_main()
{
    hpx::execution::experimental::fork_join_executor exec{};

    auto const result = hpx::transform_reduce(hpx::execution::par.on(exec),
        hpx::util::counting_iterator(0), hpx::util::counting_iterator(100), 0L,
        std::plus{}, [&](auto i) { return i * i; });

    HPX_TEST_EQ(result, 328350L);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
