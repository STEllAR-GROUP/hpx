//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/algorithm.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <mutex>
#include <set>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::lcos::local::spinlock mtx;
    std::set<hpx::thread::id> thread_ids;

    hpx::experimental::for_loop(hpx::execution::par, 0, 100, [&](int) {
        std::lock_guard<hpx::lcos::local::spinlock> l(mtx);
        thread_ids.insert(hpx::this_thread::get_id());
    });

    HPX_TEST_LT(std::size_t(1), thread_ids.size());

    thread_ids.clear();

    hpx::experimental::for_loop_n(hpx::execution::par, 0, 100, [&](int) {
        std::lock_guard<hpx::lcos::local::spinlock> l(mtx);
        thread_ids.insert(hpx::this_thread::get_id());
    });

    HPX_TEST_LT(std::size_t(1), thread_ids.size());

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=4"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
