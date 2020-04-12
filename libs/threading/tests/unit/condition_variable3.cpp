//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/hpx_main.hpp>
#include <hpx/synchronization.hpp>
#include <hpx/testing.hpp>
#include <hpx/threading.hpp>

#include <chrono>
#include <functional>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
void test_cv_callback()
{
    bool ready{false};
    hpx::lcos::local::mutex ready_mtx;
    hpx::lcos::local::condition_variable_any ready_cv;

    bool cb_called{false};
    {
        hpx::jthread t1{[&](hpx::stop_token stoken) {
            auto f = [&] {
                hpx::this_thread::sleep_for(std::chrono::seconds(1));
                cb_called = true;
            };
            hpx::stop_callback<std::function<void()>> cb(stoken, std::move(f));

            std::unique_lock<hpx::lcos::local::mutex> lg{ready_mtx};
            ready_cv.wait(lg, stoken, [&ready] { return ready; });
        }};

        hpx::this_thread::sleep_for(std::chrono::seconds(1));
    }    // leave scope of t1 without join() or detach() (signals cancellation)
    HPX_TEST(cb_called);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::set_terminate([]() { HPX_TEST(false); });
    try
    {
        test_cv_callback();
    }
    catch (...)
    {
        HPX_TEST(false);
    }
    return hpx::util::report_errors();
}
