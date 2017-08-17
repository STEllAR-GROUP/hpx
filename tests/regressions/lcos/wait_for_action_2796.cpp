//  Copyright (c) 2017 KADichev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <chrono>

boost::atomic<bool> called(false);

void f()
{
    called.store(true);
}

HPX_PLAIN_ACTION(f, f_action);
HPX_PLAIN_DIRECT_ACTION(f, f_direct_action);

int main()
{
    called.store(false);
    {
        auto fut = hpx::async<f_action>(hpx::find_here());
        auto status = fut.wait_for(std::chrono::seconds(3));
        HPX_TEST(status != hpx::lcos::future_status::deferred);
        HPX_TEST(called.load());
    }

    called.store(false);
    {
        auto fut = hpx::async<f_direct_action>(hpx::find_here());
        auto status = fut.wait_for(std::chrono::seconds(3));
        HPX_TEST(status != hpx::lcos::future_status::deferred);
        HPX_TEST(called.load());
    }

    return 0;
}
