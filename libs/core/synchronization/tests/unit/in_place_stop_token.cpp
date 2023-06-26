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

#include <chrono>
#include <utility>

void test_stop_token_basic_api()
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
    bool cb1called{false};
    auto cb1 = [&] { cb1called = true; };
    {
        hpx::experimental::in_place_stop_callback<decltype(cb1)> scb1(
            stok, cb1);    // copies cb1
        HPX_TEST(ssrc.stop_possible());
        HPX_TEST(!ssrc.stop_requested());
        HPX_TEST(stok.stop_possible());
        HPX_TEST(!stok.stop_requested());
        HPX_TEST(!cb1called);
    }    // unregister callback

    // register another callback
    bool cb2called{false};
    auto cb2 = [&] {
        HPX_TEST(stok.stop_requested());
        cb2called = true;
    };

    hpx::experimental::in_place_stop_callback<decltype(cb2)> scb2a{
        stok, cb2};    // copies cb2
    hpx::experimental::in_place_stop_callback<decltype(cb2)> scb2b{
        stok, std::move(cb2)};
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(!stok.stop_requested());
    HPX_TEST(!cb1called);
    HPX_TEST(!cb2called);

    // request stop
    auto b = ssrc.request_stop();
    HPX_TEST(b);
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(stok.stop_requested());
    HPX_TEST(!cb1called);
    HPX_TEST(cb2called);

    b = ssrc.request_stop();
    HPX_TEST(!b);

    // register another callback
    bool cb3called{false};

    auto cb3 = [&] { cb3called = true; };
    hpx::experimental::in_place_stop_callback<decltype(cb3)> scb3(
        stok, std::move(cb3));
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(stok.stop_requested());
    HPX_TEST(!cb1called);
    HPX_TEST(cb2called);
    HPX_TEST(cb3called);
}

///////////////////////////////////////////////////////////////////////////////
void test_stop_token_api()
{
    // stop_source: create and destroy
    {
        hpx::experimental::in_place_stop_source is1;
        HPX_TEST(is1.stop_possible());
    }

    // stop_token: create, copy, assign and destroy
    {
        hpx::experimental::in_place_stop_token it1;
        hpx::experimental::in_place_stop_token it2{it1};
        hpx::experimental::in_place_stop_token it3 = it1;
        hpx::experimental::in_place_stop_token it4{std::move(it1)};
        it1 = it2;
        it1 = std::move(it2);
        std::swap(it1, it2);
        it1.swap(it2);
        HPX_TEST(!it1.stop_possible());
        HPX_TEST(!it2.stop_possible());
        HPX_TEST(!it3.stop_possible());
        HPX_TEST(!it4.stop_possible());
    }

    // assignment and swap()
    {
        hpx::experimental::in_place_stop_source is_stopped;
        is_stopped.request_stop();

        hpx::experimental::in_place_stop_token it_stopped{
            is_stopped.get_token()};

        // assignments and swap()
        HPX_TEST(!hpx::experimental::in_place_stop_token{}.stop_requested());
        it_stopped = hpx::experimental::in_place_stop_token{};
        HPX_TEST(!it_stopped.stop_possible());
        HPX_TEST(!it_stopped.stop_requested());
    }

    // shared ownership semantics
    hpx::experimental::in_place_stop_source is;
    hpx::experimental::in_place_stop_token it1{is.get_token()};
    hpx::experimental::in_place_stop_token it2{it1};
    HPX_TEST(is.stop_possible() && !is.stop_requested());
    HPX_TEST(it1.stop_possible() && !it1.stop_requested());
    HPX_TEST(it2.stop_possible() && !it2.stop_requested());
    is.request_stop();
    HPX_TEST(is.stop_possible() && is.stop_requested());
    HPX_TEST(it1.stop_possible() && it1.stop_requested());
    HPX_TEST(it2.stop_possible() && it2.stop_requested());
}

///////////////////////////////////////////////////////////////////////////////
template <typename D>
void sleep(D duration)
{
    if (duration > std::chrono::milliseconds{0})
    {
        std::this_thread::sleep_for(duration);
    }
}

template <typename D>
void test_stoken(D duration)
{
    int okSteps = 0;
    try
    {
        hpx::experimental::in_place_stop_source interruptor;
        hpx::experimental::in_place_stop_token interruptee{
            interruptor.get_token()};
        ++okSteps;
        sleep(duration);    // 1
        HPX_TEST(!interruptor.stop_requested());
        HPX_TEST(!interruptee.stop_requested());

        interruptor.request_stop();    // INTERRUPT !!!
        ++okSteps;
        sleep(duration);    // 2
        HPX_TEST(interruptor.stop_requested());
        HPX_TEST(interruptee.stop_requested());

        interruptor.request_stop();
        ++okSteps;
        sleep(duration);    // 3
        HPX_TEST(interruptor.stop_requested());
        HPX_TEST(interruptee.stop_requested());

        hpx::experimental::in_place_stop_source interruptor2{};
        interruptee = interruptor2.get_token();
        ++okSteps;
        sleep(duration);    // 4
        HPX_TEST(!interruptor2.stop_requested());
        HPX_TEST(!interruptee.stop_requested());

        interruptor2.request_stop();    // INTERRUPT !!!
        ++okSteps;
        sleep(duration);    // 5
        HPX_TEST(interruptor2.stop_requested());
        HPX_TEST(interruptee.stop_requested());

        interruptor2.request_stop();
        ++okSteps;
        sleep(duration);    // 6
        HPX_TEST(interruptor2.stop_requested());
        HPX_TEST(interruptee.stop_requested());
    }
    catch (...)
    {
        HPX_TEST(false);
    }
    HPX_TEST(okSteps == 6);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_stop_token_basic_api();
    test_stop_token_api();
    test_stoken(std::chrono::seconds{0});
    test_stoken(std::chrono::milliseconds{500});

    hpx::local::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
