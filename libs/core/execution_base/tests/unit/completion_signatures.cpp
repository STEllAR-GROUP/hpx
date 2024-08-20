//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/coroutines_support.hpp>

#include <exception>
#include <utility>

namespace ex = hpx::execution::experimental;

///////////////////////////////////////////////////////////////////////////////
template <typename... Values>
auto signature_values(Values...)
    -> ex::completion_signatures<ex::set_value_t(Values...)>
{
    return {};
}

template <typename Error>
auto signature_error(Error) -> ex::completion_signatures<ex::set_error_t(Error)>
{
    return {};
}

auto signature_stopped() -> ex::completion_signatures<ex::set_stopped_t()>
{
    return {};
}

template <typename Error, typename... Values>
auto signature_error_values(Error, Values...)
    -> ex::completion_signatures<ex::set_value_t(Values...),
        ex::set_error_t(Error)>
{
    return {};
}

template <typename... Values>
auto signature_values_stopped(Values...)
    -> ex::completion_signatures<ex::set_value_t(Values...),
        ex::set_stopped_t()>
{
    return {};
}

template <typename Error>
auto signature_error_stopped(Error)
    -> ex::completion_signatures<ex::set_error_t(Error), ex::set_stopped_t()>
{
    return {};
}

template <typename Error, typename... Values>
auto signature_all(Error, Values...)
    -> ex::completion_signatures<ex::set_value_t(Values...),
        ex::set_error_t(Error), ex::set_stopped_t()>
{
    return {};
}

///////////////////////////////////////////////////////////////////////////////
template <typename... Values>
void test_values(Values...)
{
    using completion_signatures = decltype(signature_values(Values()...));

    using value_types =
        typename completion_signatures::template value_types<hpx::tuple,
            hpx::variant>;
    using error_types =
        typename completion_signatures::template error_types<hpx::variant>;

    static_assert(
        std::is_same_v<value_types, hpx::variant<hpx::tuple<Values...>>>);
    static_assert(std::is_same_v<error_types, hpx::variant<>>);
    static_assert(!completion_signatures::sends_stopped);
}

template <typename Error>
void test_error(Error)
{
    using completion_signatures = decltype(signature_error(Error()));

    using value_types =
        typename completion_signatures::template value_types<hpx::tuple,
            hpx::variant>;
    using error_types =
        typename completion_signatures::template error_types<hpx::variant>;

    static_assert(std::is_same_v<value_types, hpx::variant<>>);
    static_assert(std::is_same_v<error_types, hpx::variant<Error>>);
    static_assert(!completion_signatures::sends_stopped);
}

void test_stopped()
{
    using completion_signatures = decltype(signature_stopped());

    using value_types =
        typename completion_signatures::template value_types<hpx::tuple,
            hpx::variant>;
    using error_types =
        typename completion_signatures::template error_types<hpx::variant>;

    static_assert(std::is_same_v<value_types, hpx::variant<>>);
    static_assert(std::is_same_v<error_types, hpx::variant<>>);
    static_assert(completion_signatures::sends_stopped);
}

template <typename Error, typename... Values>
void test_error_values(Error, Values...)
{
    using completion_signatures =
        decltype(signature_error_values(Error(), Values()...));

    using value_types =
        typename completion_signatures::template value_types<hpx::tuple,
            hpx::variant>;
    using error_types =
        typename completion_signatures::template error_types<hpx::variant>;

    static_assert(
        std::is_same_v<value_types, hpx::variant<hpx::tuple<Values...>>>);
    static_assert(std::is_same_v<error_types, hpx::variant<Error>>);
    static_assert(!completion_signatures::sends_stopped);
}

template <typename... Values>
void test_values_stopped(Values...)
{
    using completion_signatures =
        decltype(signature_values_stopped(Values()...));

    using value_types =
        typename completion_signatures::template value_types<hpx::tuple,
            hpx::variant>;
    using error_types =
        typename completion_signatures::template error_types<hpx::variant>;

    static_assert(
        std::is_same_v<value_types, hpx::variant<hpx::tuple<Values...>>>);
    static_assert(std::is_same_v<error_types, hpx::variant<>>);
    static_assert(completion_signatures::sends_stopped);
}

template <typename Error>
void test_error_stopped(Error)
{
    using completion_signatures = decltype(signature_error_stopped(Error()));

    using value_types =
        typename completion_signatures::template value_types<hpx::tuple,
            hpx::variant>;
    using error_types =
        typename completion_signatures::template error_types<hpx::variant>;

    static_assert(std::is_same_v<value_types, hpx::variant<>>);
    static_assert(std::is_same_v<error_types, hpx::variant<Error>>);
    static_assert(completion_signatures::sends_stopped);
}

template <typename Error, typename... Values>
void test_all(Error, Values...)
{
    using completion_signatures = decltype(signature_all(Error(), Values()...));

    using value_types =
        typename completion_signatures::template value_types<hpx::tuple,
            hpx::variant>;
    using error_types =
        typename completion_signatures::template error_types<hpx::variant>;

    static_assert(
        std::is_same_v<value_types, hpx::variant<hpx::tuple<Values...>>>);
    static_assert(std::is_same_v<error_types, hpx::variant<Error>>);
    static_assert(completion_signatures::sends_stopped);
}

// clang-format off
template <typename Variant>
struct possibly_empty_variant
{
    using type = Variant;
};

template <>
struct possibly_empty_variant<hpx::variant<>>
{
    using type = ex::empty_variant;
};

template <typename Variant>
using possibly_empty_variant_t =
    hpx::meta::type<possibly_empty_variant<Variant>>;

// clang-format on

template <typename Signatures>
struct sender_1
{
    using completion_signatures = Signatures;
};

template <typename Signatures>
void test_sender1(Signatures)
{
    static_assert(ex::is_sender_v<sender_1<Signatures>>);

    sender_1<Signatures> s;
    static_assert(hpx::meta::value<
        ex::detail::has_completion_signatures<sender_1<Signatures>>>);
    static_assert(
        std::is_same_v<decltype(ex::get_completion_signatures(s)), Signatures>);
    static_assert(
        std::is_same_v<ex::completion_signatures_of_t<sender_1<Signatures>>,
            Signatures>);

    using value_types_of = ex::value_types_of_t<sender_1<Signatures>>;
    using value_types = possibly_empty_variant_t<
        typename Signatures::template value_types<hpx::tuple, hpx::variant>>;

    static_assert(std::is_same_v<value_types_of, value_types>);

    using error_types_of = ex::error_types_of_t<sender_1<Signatures>>;
    using error_types = possibly_empty_variant_t<
        typename Signatures::template error_types<hpx::variant>>;

    static_assert(std::is_same_v<error_types_of, error_types>);
}

template <typename Signatures>
struct sender_2
{
};

template <typename Signatures, typename Env = ex::no_env>
constexpr auto tag_invoke(ex::get_completion_signatures_t,
    sender_2<Signatures> const&, Env = Env{}) noexcept -> Signatures
{
    return {};
}

template <typename Signatures>
void test_sender2(Signatures)
{
    {
        static_assert(ex::is_sender_v<sender_2<Signatures>>);

        sender_2<Signatures> s1;
        static_assert(
            hpx::functional::is_tag_invocable_v<ex::get_completion_signatures_t,
                sender_2<Signatures>>);
        static_assert(
            std::is_same_v<decltype(ex::get_completion_signatures(s1)),
                Signatures>);
        static_assert(
            std::is_same_v<ex::completion_signatures_of_t<sender_2<Signatures>>,
                Signatures>);

        using value_types_of = ex::value_types_of_t<sender_2<Signatures>>;
        using value_types =
            possibly_empty_variant_t<typename Signatures::template value_types<
                hpx::tuple, hpx::variant>>;

        static_assert(std::is_same_v<value_types_of, value_types>);

        using error_types_of = ex::error_types_of_t<sender_2<Signatures>>;
        using error_types = possibly_empty_variant_t<
            typename Signatures::template error_types<hpx::variant>>;

        static_assert(std::is_same_v<error_types_of, error_types>);
    }
    {
        static_assert(ex::is_sender_v<sender_2<Signatures>, ex::no_env>);

        sender_2<Signatures> s2;
        static_assert(
            hpx::functional::is_tag_invocable_v<ex::get_completion_signatures_t,
                sender_2<Signatures>, ex::no_env>);
        static_assert(
            std::is_same_v<decltype(ex::get_completion_signatures(s2)),
                Signatures>);
        static_assert(std::is_same_v<
            ex::completion_signatures_of_t<sender_2<Signatures>, ex::no_env>,
            Signatures>);

        using value_types_of =
            ex::value_types_of_t<sender_2<Signatures>, ex::no_env>;
        using value_types =
            possibly_empty_variant_t<typename Signatures::template value_types<
                hpx::tuple, hpx::variant>>;

        static_assert(std::is_same_v<value_types_of, value_types>);

        using error_types_of =
            ex::error_types_of_t<sender_2<Signatures>, ex::no_env>;
        using error_types = possibly_empty_variant_t<
            typename Signatures::template error_types<hpx::variant>>;

        static_assert(std::is_same_v<error_types_of, error_types>);
    }
}

struct sender_3
{
};

void test_sender3()
{
    sender_3 s;
    static_assert(std::is_same_v<decltype(ex::get_completion_signatures(s)),
        ex::detail::no_completion_signatures>);
}

#if defined(HPX_HAVE_CXX20_COROUTINES)

template <typename Awaiter>
struct promise
{
    hpx::coroutine_handle<promise> get_return_object()
    {
        return {hpx::coroutine_handle<promise>::from_promise(*this)};
    }
    hpx::suspend_always initial_suspend() noexcept
    {
        return {};
    }
    hpx::suspend_always final_suspend() noexcept
    {
        return {};
    }
    void return_void() {}
    void unhandled_exception() {}

    template <typename... T>
    auto await_transform(T&&...) noexcept
    {
        return std::declval<Awaiter>();
    }
};

struct awaiter
{
    void await_ready() {}
    bool await_suspend(hpx::coroutine_handle<>)
    {
        return false;
    }
    bool await_resume()
    {
        return false;
    }
};

template <typename Awaiter>
struct awaitable_sender_1
{
    Awaiter operator co_await();
};

struct awaitable_sender_2
{
    using promise_type = promise<hpx::suspend_always>;
};

struct awaitable_sender_3
{
    using promise_type = promise<awaiter>;
};

template <typename Signatures, typename Awaiter>
void test_awaitable_sender1(Signatures&&, Awaiter&&)
{
    static_assert(ex::is_sender_v<awaitable_sender_1<Awaiter>>);
    static_assert(ex::is_awaitable_v<awaitable_sender_1<Awaiter>>);

    awaitable_sender_1<Awaiter> s;
    static_assert(!hpx::meta::value<ex::detail::has_completion_signatures<
                      awaitable_sender_1<Awaiter>>>);
    static_assert(
        std::is_same_v<decltype(ex::get_completion_signatures(s)), Signatures>);
    static_assert(std::is_same_v<
        ex::completion_signatures_of_t<awaitable_sender_1<Awaiter>>,
        Signatures>);

    using value_types_of = ex::value_types_of_t<awaitable_sender_1<Awaiter>>;
    using value_types = possibly_empty_variant_t<
        typename Signatures::template value_types<hpx::tuple, hpx::variant>>;

    static_assert(std::is_same_v<value_types_of, value_types>);

    using error_types_of = ex::error_types_of_t<awaitable_sender_1<Awaiter>>;
    using error_types = possibly_empty_variant_t<
        typename Signatures::template error_types<hpx::variant>>;

    static_assert(std::is_same_v<error_types_of, error_types>);
}

template <typename Signatures>
void test_awaitable_sender2(Signatures)
{
    // is_sender_v relies on get_completion_signatures and is not true
    // even if a sender is an awaitable if it's promise type is not
    // used to evaluate the concept awaitable
    // static_assert(ex::is_sender_v<awaitable_sender_2>);

    // static_assert(ex::is_awaitable_v<awaitable_sender_2,
    //     promise<hpx::suspend_always>>);

    // awaitable_sender_2 s;
    // static_assert(!hpx::meta::value<
    //               ex::detail::has_completion_signatures<awaitable_sender_2>>);

    // static_assert(
    //     std::is_same_v<decltype(ex::get_completion_signatures(s)), Signatures>);
    // static_assert(
    //     std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_2>,
    //         Signatures>);

    // using value_types_of = ex::value_types_of_t<awaitable_sender_2>;
    // using value_types = possibly_empty_variant_t<
    //     typename Signatures::template value_types<hpx::tuple, hpx::variant>>;

    // static_assert(std::is_same_v<value_types_of, value_types>);

    // using error_types_of = ex::error_types_of_t<awaitable_sender_2>;
    // using error_types = possibly_empty_variant_t<
    //     typename Signatures::template error_types<hpx::variant>>;

    // static_assert(std::is_same_v<error_types_of, error_types>);
}

template <typename Signatures>
void test_awaitable_sender3(Signatures)
{
    // is_sender_v relies on get_completion_signatures and is not true
    // even if a sender is an awaitable if it's promise type is not
    // used to evaluate the concept awaitable
    // static_assert(ex::is_sender_v<awaitable_sender_3>);
    // static_assert(ex::is_awaitable_v<awaitable_sender_3, promise<awaiter>>);

    // awaitable_sender_3 s;
    // static_assert(hpx::meta::value<
    //     ex::detail::has_completion_signatures<awaitable_sender_3>>);
    // static_assert(
    //     std::is_same_v<decltype(ex::get_completion_signatures(s)), Signatures>);
    // static_assert(
    //     std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_3>,
    //         Signatures>);

    // using value_types_of = ex::value_types_of_t<awaitable_sender_3>;
    // using value_types = possibly_empty_variant_t<
    //     typename Signatures::template value_types<hpx::tuple, hpx::variant>>;

    // static_assert(std::is_same_v<value_types_of, value_types>);

    // using error_types_of = ex::error_types_of_t<awaitable_sender_3>;
    // using error_types = possibly_empty_variant_t<
    //     typename Signatures::template error_types<hpx::variant>>;

    // static_assert(std::is_same_v<error_types_of, error_types>);
}

#endif    // HPX_HAVE_CXX20_COROUTINES

int main()
{
    {
        test_values();
        test_values(int());
        test_values(int(), double());

        test_error(std::exception_ptr());

        test_stopped();

        test_error_values(std::exception_ptr());
        test_error_values(std::exception_ptr(), int());
        test_error_values(std::exception_ptr(), int(), double());

        test_values_stopped();
        test_values_stopped(int());
        test_values_stopped(int(), double());

        test_error_stopped(std::exception_ptr());

        test_all(std::exception_ptr());
        test_all(std::exception_ptr(), int());
        test_all(std::exception_ptr(), int(), double());
    }
    {
        test_sender1(signature_values());
        test_sender1(signature_values(int()));
        test_sender1(signature_values(int(), double()));

        test_sender1(signature_error(std::exception_ptr()));

        test_sender1(signature_stopped());

        test_sender1(signature_error_values(std::exception_ptr()));
        test_sender1(signature_error_values(std::exception_ptr(), int()));
        test_sender1(
            signature_error_values(std::exception_ptr(), int(), double()));

        test_sender1(signature_values_stopped());
        test_sender1(signature_values_stopped(int()));
        test_sender1(signature_values_stopped(int(), double()));

        test_sender1(signature_error_stopped(std::exception_ptr()));

        test_sender1(signature_all(std::exception_ptr()));
        test_sender1(signature_all(std::exception_ptr(), int()));
        test_sender1(signature_all(std::exception_ptr(), int(), double()));
    }
    {
        test_sender2(signature_values());
        test_sender2(signature_values(int()));
        test_sender2(signature_values(int(), double()));

        test_sender2(signature_error(std::exception_ptr()));

        test_sender2(signature_stopped());

        test_sender2(signature_error_values(std::exception_ptr()));
        test_sender2(signature_error_values(std::exception_ptr(), int()));
        test_sender2(
            signature_error_values(std::exception_ptr(), int(), double()));

        test_sender2(signature_values_stopped());
        test_sender2(signature_values_stopped(int()));
        test_sender2(signature_values_stopped(int(), double()));

        test_sender2(signature_error_stopped(std::exception_ptr()));

        test_sender2(signature_all(std::exception_ptr()));
        test_sender2(signature_all(std::exception_ptr(), int()));
        test_sender2(signature_all(std::exception_ptr(), int(), double()));
    }

    test_sender3();

#if defined(HPX_HAVE_CXX20_COROUTINES)

    {
        test_awaitable_sender1(signature_error_values(std::exception_ptr()),
            hpx::suspend_always{});
        test_awaitable_sender1(
            signature_error_values(std::exception_ptr(), bool()), awaiter{});

        // TODO: handle for awaitables that do not have co_await free/member
        // operator
        // test_awaitable_sender2(signature_error_values(std::exception_ptr()));
        // test_awaitable_sender3(signature_error_values(std::exception_ptr()));
    }

#endif    // HPX_HAVE_CXX20_COROUTINES

    return hpx::util::report_errors();
}
