//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <functional>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
void test_cv_callback()
{
    bool ready{false};
    hpx::mutex ready_mtx;
    hpx::condition_variable_any ready_cv;

    bool cb_called{false};
    {
        hpx::jthread t1{[&](hpx::stop_token stoken) {
            auto f = [&] {
                hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
                cb_called = true;
            };
            hpx::stop_callback<std::function<void()>> cb(stoken, std::move(f));

            std::unique_lock<hpx::mutex> lg{ready_mtx};
            ready_cv.wait(lg, stoken, [&ready] { return ready; });
        }};

        hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    }    // leave scope of t1 without join() or detach() (signals cancellation)
    HPX_TEST(cb_called);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
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

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
