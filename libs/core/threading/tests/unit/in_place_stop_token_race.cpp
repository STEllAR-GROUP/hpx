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
#include <hpx/modules/testing.hpp>
#include <hpx/optional.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <thread>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
void test_callback_register()
{
    // create stop_source
    hpx::experimental::in_place_stop_source ssrc;
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());

    // create stop_token from stop_source
    hpx::experimental::in_place_stop_token stok{ssrc.get_token()};
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(!stok.stop_requested());

    // register callback
    bool cb1_called{false};
    bool cb2_called{false};

    auto cb = [&] {
        cb1_called = true;
        // register another callback while callbacks are being executed
        auto f = [&] { cb2_called = true; };
        hpx::experimental::in_place_stop_callback<std::function<void()>> cb2(
            stok, std::move(f));
    };

    hpx::experimental::in_place_stop_callback<decltype(cb)> cb1(stok, cb);
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(!stok.stop_requested());
    HPX_TEST(!cb1_called);
    HPX_TEST(!cb2_called);

    // request stop
    auto b = ssrc.request_stop();
    HPX_TEST(b);
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(stok.stop_requested());
    HPX_TEST(cb1_called);
    HPX_TEST(cb2_called);
}

///////////////////////////////////////////////////////////////////////////////
void test_callback_unregister()
{
    // create stop_source
    hpx::experimental::in_place_stop_source ssrc;
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());

    // create stop_token from stop_source
    hpx::experimental::in_place_stop_token stok{ssrc.get_token()};
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(!stok.stop_requested());

    // register callback that unregisters itself
    bool cb1_called = false;
    hpx::optional<
        hpx::experimental::in_place_stop_callback<std::function<void()>>>
        cb;
    cb.emplace(stok, [&] {
        cb1_called = true;
        // remove this lambda in optional while being called
        cb.reset();
    });

    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(!stok.stop_requested());
    HPX_TEST(!cb1_called);

    // request stop
    auto b = ssrc.request_stop();
    HPX_TEST(b);
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(stok.stop_requested());
    HPX_TEST(cb1_called);
}

///////////////////////////////////////////////////////////////////////////////
struct reg_unreg_cb
{
    hpx::optional<
        hpx::experimental::in_place_stop_callback<std::function<void()>>>
        cb{};
    bool called = false;

    void reg(hpx::experimental::in_place_stop_token& stok)
    {
        cb.emplace(stok, [&] { called = true; });
    }
    void unreg()
    {
        cb.reset();
    }
};

void test_callback_concurrent_unregister()
{
    // create stop_source and stop_token:
    hpx::experimental::in_place_stop_source ssrc;
    hpx::experimental::in_place_stop_token stok{ssrc.get_token()};

    std::atomic<bool> cb1_called{false};
    hpx::optional<
        hpx::experimental::in_place_stop_callback<std::function<void()>>>
        opt_cb;

    auto cb1 = [&] {
        opt_cb.reset();
        cb1_called = true;
    };

    opt_cb.emplace(stok, std::ref(cb1));

    // request stop
    ssrc.request_stop();

    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(stok.stop_requested());

    HPX_TEST(cb1_called);
}

///////////////////////////////////////////////////////////////////////////////
void test_callback_concurrent_unregister_other_thread()
{
    // create stop_source and stop_token:
    hpx::experimental::in_place_stop_source ssrc;
    hpx::experimental::in_place_stop_token stok{ssrc.get_token()};

    std::atomic<bool> cb1_called{false};
    hpx::optional<
        hpx::experimental::in_place_stop_callback<std::function<void()>>>
        opt_cb;

    auto cb1 = [&] {
        opt_cb.reset();
        cb1_called = true;
    };

    hpx::thread t{[&] { opt_cb.emplace(stok, std::ref(cb1)); }};

    // request stop
    ssrc.request_stop();

    t.join();

    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(stok.stop_requested());

    HPX_TEST(cb1_called);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_callback_register();
    test_callback_unregister();

    test_callback_concurrent_unregister();
    test_callback_concurrent_unregister_other_thread();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
