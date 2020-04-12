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
#include <hpx/testing.hpp>
#include <hpx/threading.hpp>

#include <functional>
#include <type_traits>
#include <utility>

void test_stop_callback_inits()
{
    hpx::stop_token token;

    struct implicit_arg
    {
    };
    struct explicit_arg
    {
    };

    struct my_callback
    {
        my_callback() {}
        my_callback(implicit_arg) {}
        explicit my_callback(explicit_arg) {}

        void operator()()
        {
            HPX_TEST(false);
        }
    };

    auto stop10 = [] {};
    hpx::stop_callback<decltype(stop10)> cb10{token, stop10};
    static_assert(
        std::is_same_v<decltype(cb10)::callback_type, decltype(stop10)>);

    auto stop11 = [] { HPX_TEST(false); };
    hpx::stop_callback<decltype(stop11)> cb11{token, std::move(stop11)};
    static_assert(
        std::is_same_v<decltype(cb11)::callback_type, decltype(stop11)>);
    static_assert(!std::is_reference_v<decltype(cb11)::callback_type>);

    auto stop12 = [] { HPX_TEST(false); };
    hpx::stop_callback<std::reference_wrapper<decltype(stop12)>> cb12{
        token, std::ref(stop12)};
    static_assert(std::is_same_v<decltype(cb12)::callback_type,
        std::reference_wrapper<decltype(stop12)>>);
    static_assert(!std::is_reference_v<decltype(cb12)::callback_type>);

    auto cb13 = hpx::make_stop_callback(token, [] { HPX_TEST(false); });
    static_assert(!std::is_reference_v<decltype(cb13)::callback_type>);

    hpx::stop_callback<std::function<void()>> cb14{
        token, [] { HPX_TEST(false); }};
    static_assert(
        std::is_same_v<decltype(cb14)::callback_type, std::function<void()>>);
    static_assert(!std::is_reference_v<decltype(cb14)::callback_type>);

    std::function<void()> stop15 = [] { HPX_TEST(false); };
    auto cb15 = hpx::make_stop_callback(token, stop15);
    static_assert(
        std::is_same_v<decltype(cb15)::callback_type, std::function<void()>>);
    static_assert(!std::is_reference_v<decltype(cb15)::callback_type>);

    std::function<void()> stop16 = [] { HPX_TEST(false); };
    hpx::stop_callback<std::function<void()>> cb16{token, stop16};
    static_assert(
        std::is_same_v<decltype(cb16)::callback_type, std::function<void()>>);
    static_assert(!std::is_reference_v<decltype(cb16)::callback_type>);

    std::function<void()> stop17;
    auto cb17 = [token] {
        std::function<void()> f;
        if (true)
        {
            f = [] { HPX_TEST(false); };
        }
        else
        {
            f = [] { HPX_TEST(false); };
        }
        return hpx::make_stop_callback(token, f);
    }();
    static_assert(
        std::is_same_v<decltype(cb17)::callback_type, std::function<void()>>);
    static_assert(!std::is_reference_v<decltype(cb17)::callback_type>);

    implicit_arg i;
    hpx::stop_callback<my_callback> cb18{token, i};
    static_assert(std::is_same_v<decltype(cb18)::callback_type, my_callback>);
    static_assert(!std::is_reference_v<decltype(cb18)::callback_type>);

    explicit_arg e;
    hpx::stop_callback<my_callback> cb19{token, e};
    static_assert(std::is_same_v<decltype(cb19)::callback_type, my_callback>);
    static_assert(!std::is_reference_v<decltype(cb19)::callback_type>);

    // the following should fail compiling
    // hpx::stop_callback<my_callback> cb =
    //   []() -> stop_callback<my_callback> {
    //       explicit_arg e;
    //       return {token, e};
    //   }();
    //
    // hpx::stop_callback<my_callback> cb =
    //   []() -> stop_callback<my_callback> {
    //       implicit_arg i;
    //       return {token, i};
    //   }();
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    try
    {
        test_stop_callback_inits();
    }
    catch (...)
    {
        HPX_TEST(false);
    }
    return hpx::util::report_errors();
}
