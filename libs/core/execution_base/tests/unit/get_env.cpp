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

#include <iostream>

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

#if defined(HPX_HAVE_STDEXEC)
        decltype(auto) get_env() const noexcept
#else
        friend some_env tag_invoke(ex::get_env_t, receiver_2 const&) noexcept
#endif
        {
            return some_env{};
        }
    };

#if defined(HPX_HAVE_STDEXEC)
    inline constexpr struct receiver_env_t final : ex::forwarding_query_t
    {
        // clang-format off
        template <typename Env>
        decltype(auto) operator()(Env const& e) const noexcept
        {
            return e.query(*this);
        }
        // clang-format on
    } receiver_env{};
#else
    inline constexpr struct receiver_env_t final
      : hpx::functional::tag<receiver_env_t>
    {
    } receiver_env{};
#endif

#if defined(HPX_HAVE_STDEXEC)
    auto env3 = ex::prop(receiver_env, 42);
    using env3_t = decltype(env3);
#else
    using env3_t = ex::make_env_t<receiver_env_t, int>;
#endif

    struct receiver_3
    {
        using is_receiver = void;

#if defined(HPX_HAVE_STDEXEC)
        decltype(auto) get_env() const noexcept
        {
            auto f = ex::prop(receiver_env, 42);
            return f;
        }
#else
        friend constexpr auto tag_invoke(
            ex::get_env_t, receiver_3 const&) noexcept
        {
            return ex::make_env<receiver_env_t>(42);
        }
#endif
    };

#if defined(HPX_HAVE_STDEXEC)
    // clang-format off
    auto env4 = ex::env(
        std::move(env3), ex::prop(receiver_env, std::string("42")));
    // clang-format on
    using env4_t = decltype(env4);
#else
    using env4_t = ex::make_env_t<receiver_env_t, std::string, env3_t>;
#endif

    struct receiver_4
    {
        using is_receiver = void;

#if defined(HPX_HAVE_STDEXEC)
        decltype(auto) get_env() const noexcept
        {
            receiver_3 rcv;

            /* Due to https://github.com/llvm/llvm-project/issues/88077
            * The following line never compiles, and if it does, it misbehaves.
            * return ex::env(
            *           ex::prop(receiver_env, std::string("42")),
            *           ex::get_env(rcv)
            *       );
            *
            * The following is a workaround */

            auto e = ex::get_env(rcv);
            auto p = ex::prop(receiver_env, std::string("42"));

            return ex::env(std::move(e), std::move(p));
        }
#else
        friend auto tag_invoke(ex::get_env_t, receiver_4 const&) noexcept
        {
            receiver_3 rcv;

            return ex::make_env<receiver_env_t>(
                std::string("42"), ex::get_env(std::move(rcv)));
        }
#endif
    };

#if defined(HPX_HAVE_STDEXEC)
    inline constexpr struct receiver_env1_t final : ex::forwarding_query_t
    {
        template <typename Env>
        decltype(auto) operator()(Env const& e) const noexcept
        {
            return e.query(*this);
        }
    } receiver_env1{};
#else
    inline constexpr struct receiver_env1_t final
      : hpx::functional::tag<receiver_env1_t>
    {
    } receiver_env1{};
#endif

#if defined(HPX_HAVE_STDEXEC)
    // clang-format off
    auto env5 =
        ex::env(std::move(env3), ex::prop(receiver_env1, std::string("42")));
    // clang-format on
    using env5_t = decltype(env5);
#else
    using env5_t = ex::make_env_t<receiver_env1_t, std::string, env3_t>;
#endif

    struct receiver_5
    {
        using is_receiver = void;

#if defined(HPX_HAVE_STDEXEC)
        decltype(auto) get_env() const noexcept
        {
            receiver_3 rcv;
            /* Same as receiver_4
             * This would cause the compiler to crash:
             * return ex::env(
             *    ex::get_env(rcv), ex::prop(receiver_env1, std::string("42")));
             * */
            auto e = ex::get_env(rcv);
            auto p = ex::prop(receiver_env1, std::string("42"));
            return ex::env(std::move(e), std::move(p));
        }
#else
        friend auto tag_invoke(ex::get_env_t, receiver_5 const&) noexcept
        {
            receiver_3 rcv;
            return ex::make_env<receiver_env1_t>(
                std::string("42"), ex::get_env(std::move(rcv)));
        }
#endif
    };
}    // namespace mylib

int main()
{
    using ex::empty_env;
#if !defined(HPX_HAVE_STDEXEC)
    using ex::is_no_env_v;
    using ex::no_env;

    static_assert(
        is_no_env_v<no_env>, "no_env is a (possibly cv-qualified) no_env");
    static_assert(is_no_env_v<no_env const&>,
        "no_env is a (possibly cv-qualified) no_env");
    static_assert(!is_no_env_v<some_env>,
        "some_env is not a (possibly cv-qualified) no_env");
#endif

    {
        mylib::receiver_1 rcv;
#if defined(HPX_HAVE_STDEXEC)
        auto env = ex::get_env(rcv);
#else
        auto env = ex::get_env(std::move(rcv));
#endif
        static_assert(
            std::is_same_v<decltype(env), empty_env>, "must return empty_env");
    }
    {
        mylib::receiver_2 rcv;
#if defined(HPX_HAVE_STDEXEC)
        auto env = ex::get_env(rcv);
#else
        auto env = ex::get_env(std::move(rcv));
#endif
        static_assert(
            std::is_same_v<decltype(env), some_env>, "must return some_env");
    }
    {
        mylib::receiver_3 rcv;
#if defined(HPX_HAVE_STDEXEC)
        auto env = ex::get_env(rcv);
#else
        auto env = ex::get_env(std::move(rcv));
#endif
        HPX_TEST_EQ(mylib::receiver_env(env), 42);
    }
    {
        mylib::receiver_4 rcv;
#if defined(HPX_HAVE_STDEXEC)
        // The resulting env_ = env(env1, env2) will query env1 first and env2
        // in that order, in order to find the resulting value of some query.
        // In cases when both env1 and env2 support the same query, as seen here
        // with receiver_env the result of env1 is picked.
        auto env = ex::get_env(rcv);
        HPX_TEST_EQ(mylib::receiver_env(env), 42);
#else
        auto env = ex::get_env(std::move(rcv));
        HPX_TEST_EQ(mylib::receiver_env(env), std::string("42"));
#endif
    }
    {
        mylib::receiver_5 rcv;
#if defined(HPX_HAVE_STDEXEC)
        auto env = ex::get_env(rcv);
#else
        auto env = ex::get_env(std::move(rcv));
#endif
        HPX_TEST_EQ(mylib::receiver_env(env), 42);
        HPX_TEST_EQ(mylib::receiver_env1(env), std::string("42"));
    }

    return hpx::util::report_errors();
}
