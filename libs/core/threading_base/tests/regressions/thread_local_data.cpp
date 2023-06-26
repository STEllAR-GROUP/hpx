//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>

std::atomic<bool> data_deallocated(false);

struct test_data
{
    test_data() = default;

    ~test_data()
    {
        data_deallocated = true;
    }
};

void test()
{
    hpx::threads::thread_id_type id = hpx::threads::get_self_id();
    test_data* p = new test_data;
    hpx::threads::add_thread_exit_callback(id, [p, id]() {
        hpx::threads::thread_id_type id1 = hpx::threads::get_self_id();
        HPX_TEST_EQ(id1, id);

        test_data* p1 =
            reinterpret_cast<test_data*>(hpx::threads::get_thread_data(id1));
        HPX_TEST_EQ(p1, p);

        delete p;
    });
    hpx::threads::set_thread_data(id, reinterpret_cast<std::size_t>(p));
}

int hpx_main()
{
    hpx::async(&test).get();
    HPX_TEST(data_deallocated);
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
