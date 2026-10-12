//  Copyright (c) 2026 John Sorial
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test for #5919: the launch::async overloads of the distributed
// channel client (get, set, close) used to block until the id of the
// referenced channel was resolved (for instance while waiting for
// hpx::find_from_basename). This could deadlock applications that issue
// asynchronous channel operations before performing other (blocking) work
// the id resolution depends on. Verify that these operations return
// immediately and are executed once the id becomes available.

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>

HPX_REGISTER_CHANNEL(int)
HPX_REGISTER_CHANNEL(void)

void test_channel_int()
{
    hpx::distributed::channel<int> const target(hpx::find_here());

    // the id of this client becomes available only after p is made ready
    hpx::promise<hpx::id_type> p;
    hpx::distributed::channel<int> c(p.get_future());
    HPX_TEST(!c.is_ready());

    // none of these may block as the id of the channel is not known yet
    hpx::future<void> set_f = c.set(hpx::launch::async, 42, 1);
    hpx::future<int> get_f = c.get(hpx::launch::async, 1);

    HPX_TEST(!set_f.is_ready());
    HPX_TEST(!get_f.is_ready());

    p.set_value(target.get_id());

    set_f.get();
    HPX_TEST_EQ(get_f.get(), 42);

    hpx::future<std::size_t> close_f = c.close(hpx::launch::async);
    HPX_TEST_EQ(close_f.get(), std::size_t(0));
}

void test_channel_void()
{
    hpx::distributed::channel<> const target(hpx::find_here());

    hpx::promise<hpx::id_type> p;
    hpx::distributed::channel<> c(p.get_future());
    HPX_TEST(!c.is_ready());

    hpx::future<void> set_f = c.set(hpx::launch::async, 1);
    hpx::future<void> get_f = c.get(hpx::launch::async, 1);

    HPX_TEST(!set_f.is_ready());
    HPX_TEST(!get_f.is_ready());

    p.set_value(target.get_id());

    set_f.get();
    get_f.get();
}

int hpx_main()
{
    test_channel_int();
    test_channel_void();

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    HPX_TEST_EQ(0, hpx::init(argc, argv));
    return hpx::util::report_errors();
}
#endif
