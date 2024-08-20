//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

struct some_env
{
};

namespace mylib {
    struct receiver_1
    {
    };

    struct receiver_2
    {
        friend some_env tag_invoke(
            hpx::execution::experimental::get_env_t, receiver_2&&)
        {
            return some_env{};
        }
    };

    inline constexpr struct receiver_env_t final
      : hpx::functional::tag<receiver_env_t>
    {
    } receiver_env{};

    using env3_t =
        hpx::execution::experimental::make_env_t<receiver_env_t, int>;

    struct receiver_3
    {
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::get_env_t, receiver_3&&) noexcept
        {
            return hpx::execution::experimental::make_env<receiver_env_t>(42);
        }
    };

    using env4_t = hpx::execution::experimental::make_env_t<receiver_env_t,
        std::string, env3_t>;

    struct receiver_4
    {
        friend auto tag_invoke(
            hpx::execution::experimental::get_env_t, receiver_4&&) noexcept
        {
            receiver_3 rcv;
            return hpx::execution::experimental::make_env<receiver_env_t>(
                std::string("42"),
                hpx::execution::experimental::get_env(std::move(rcv)));
        }
    };

    inline constexpr struct receiver_env1_t final
      : hpx::functional::tag<receiver_env1_t>
    {
    } receiver_env1{};

    using env5_t = hpx::execution::experimental::make_env_t<receiver_env1_t,
        std::string, env3_t>;

    struct receiver_5
    {
        friend auto tag_invoke(
            hpx::execution::experimental::get_env_t, receiver_5&&) noexcept
        {
            receiver_3 rcv;
            return hpx::execution::experimental::make_env<receiver_env1_t>(
                std::string("42"),
                hpx::execution::experimental::get_env(std::move(rcv)));
        }
    };
}    // namespace mylib

int main()
{
    using hpx::execution::experimental::empty_env;
    using hpx::execution::experimental::is_no_env_v;
    using hpx::execution::experimental::no_env;

    static_assert(
        is_no_env_v<no_env>, "no_env is a (possibly cv-qualified) no_env");
    static_assert(is_no_env_v<no_env const&>,
        "no_env is a (possibly cv-qualified) no_env");
    static_assert(!is_no_env_v<some_env>,
        "some_env is not a (possibly cv-qualified) no_env");

    {
        mylib::receiver_1 rcv;
        auto env = hpx::execution::experimental::get_env(std::move(rcv));
        static_assert(
            std::is_same_v<decltype(env), empty_env>, "must return empty_env");
    }
    {
        mylib::receiver_2 rcv;
        auto env = hpx::execution::experimental::get_env(std::move(rcv));
        static_assert(
            std::is_same_v<decltype(env), some_env>, "must return some_env");
    }
    {
        mylib::receiver_3 rcv;
        auto env = hpx::execution::experimental::get_env(std::move(rcv));
        HPX_TEST_EQ(mylib::receiver_env(env), 42);
    }
    {
        mylib::receiver_4 rcv;
        auto env = hpx::execution::experimental::get_env(std::move(rcv));
        HPX_TEST_EQ(mylib::receiver_env(env), std::string("42"));
    }
    {
        mylib::receiver_5 rcv;
        auto env = hpx::execution::experimental::get_env(std::move(rcv));
        HPX_TEST_EQ(mylib::receiver_env(env), 42);
        HPX_TEST_EQ(mylib::receiver_env1(env), std::string("42"));
    }

    return hpx::util::report_errors();
}
