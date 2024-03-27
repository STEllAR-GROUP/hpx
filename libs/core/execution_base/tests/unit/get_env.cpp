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

namespace ex = hpx::execution::experimental;

struct some_env
{
};

namespace mylib {
    struct receiver_1
    {
        using is_receiver = void;
    };

    struct receiver_2
    {
        using is_receiver = void;

        friend some_env tag_invoke(
            ex::get_env_t, receiver_2 const&) noexcept
        {
            return some_env{};
        }
    };

    inline constexpr struct receiver_env_t final
      : hpx::functional::tag<receiver_env_t>
    {
    } receiver_env{};

    using env3_t =
        ex::make_env_t<receiver_env_t, int>;

    struct receiver_3
    {
        using is_receiver = void;

        friend constexpr auto tag_invoke(
            ex::get_env_t, receiver_3 const&) noexcept
        {
#ifdef HPX_HAVE_STDEXEC
            return ex::make_env(ex::with(receiver_env, 42));
#else
            return ex::make_env<receiver_env_t>(42);
#endif
        }
    };

#ifdef HPX_HAVE_STDEXEC
#else
    using env4_t = ex::make_env_t<receiver_env_t,
        std::string, env3_t>;
#endif

    struct receiver_4
    {
        using is_receiver = void;

        friend auto tag_invoke(
            ex::get_env_t, receiver_4 const&) noexcept
        {
            receiver_3 rcv;

#ifdef HPX_HAVE_STDEXEC
            auto k = ex::make_env(
                ex::get_env(rcv),
                ex::with(receiver_env, std::string("42"))
                );
#else
            return ex::make_env<receiver_env_t>(
                std::string("42"),
                ex::get_env(std::move(rcv)));
#endif
        }
    };

    inline constexpr struct receiver_env1_t final
      : hpx::functional::tag<receiver_env1_t>
    {
    } receiver_env1{};

#ifdef HPX_HAVE_STDEXEC
#else
    using env5_t = ex::make_env_t<receiver_env1_t,
        std::string, env3_t>;
#endif

    struct receiver_5
    {
        using is_receiver = void;

        friend auto tag_invoke(
            ex::get_env_t, receiver_5 const&) noexcept
        {
            receiver_3 rcv;
#ifdef HPX_HAVE_STDEXEC
//            static_assert(
//                std::same_as<decltype(ex::get_env(rcv)), int>
//                );
            auto k =  ex::make_env(
                ex::get_env(rcv),
                ex::with(receiver_env1, std::string("42"))
                );

            auto n = receiver_env(k);
            return k;
#else
            return ex::make_env<receiver_env1_t>(
                std::string("42"),
                ex::get_env(std::move(rcv)));
#endif
        }
    };
}    // namespace mylib

int main()
{
    using ex::empty_env;
    using ex::is_no_env_v;
    using ex::no_env;

    static_assert(
        is_no_env_v<no_env>, "no_env is a (possibly cv-qualified) no_env");
    static_assert(is_no_env_v<no_env const&>,
        "no_env is a (possibly cv-qualified) no_env");
    static_assert(!is_no_env_v<some_env>,
        "some_env is not a (possibly cv-qualified) no_env");

    {
        mylib::receiver_1 rcv;
        auto env = ex::get_env(std::move(rcv));
        static_assert(
            std::is_same_v<decltype(env), empty_env>, "must return empty_env");
    }
    {
        mylib::receiver_2 rcv;
        auto env = ex::get_env.operator()<false>(std::move(rcv));
        static_assert(
            std::is_same_v<decltype(env), some_env>, "must return some_env");
    }
    {
        mylib::receiver_3 rcv;
        auto env = ex::get_env(std::move(rcv));
        HPX_TEST_EQ(mylib::receiver_env(env), 42);
    }
    {
        mylib::receiver_4 rcv;
        auto env = ex::get_env(rcv);
        HPX_TEST_EQ(mylib::receiver_env(env), std::string("42"));
    }
    {
        mylib::receiver_5 rcv;
        auto env = ex::get_env(rcv);
        HPX_TEST_EQ(mylib::receiver_env(env), 42);
        HPX_TEST_EQ(mylib::receiver_env1(env), std::string("42"));
    }

    return hpx::util::report_errors();
}
