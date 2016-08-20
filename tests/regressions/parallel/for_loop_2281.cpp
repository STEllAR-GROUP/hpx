//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <mutex>
#include <set>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::lcos::local::spinlock mtx;
    std::set<hpx::thread::id> thread_ids;

    hpx::parallel::for_loop(
        hpx::parallel::par, 0, 100,
        [&](int i)
        {
            std::lock_guard<hpx::lcos::local::spinlock> l(mtx);
            thread_ids.insert(hpx::this_thread::get_id());
        });

    HPX_TEST(thread_ids.size() > std::size_t(1));

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {
        "hpx.os_threads=4"
    };

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
