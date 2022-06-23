//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

static std::size_t friend_tag_invoke_connect_calls = 0;
static std::size_t tag_invoke_connect_calls = 0;

struct non_sender_1
{
};

struct non_sender_2
{
    struct completion_signatures
    {
        template <template <class...> class Variant>
        using error_types = Variant<>;

        static constexpr bool sends_stopped = false;
    };
};

struct non_sender_3
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        static constexpr bool sends_stopped = false;
    };
};

struct non_sender_4
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<>;
    };
};

struct non_sender_5
{
    struct completion_signatures
    {
        static constexpr bool sends_stopped = false;
    };
};

struct non_sender_6
{
    struct completion_signatures
    {
        template <template <class...> class Variant>
        using error_types = Variant<>;
    };
};

struct non_sender_7
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;
    };
};

struct receiver
{
    friend void tag_invoke(hpx::execution::experimental::set_error_t,
        receiver&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_invoke(
        hpx::execution::experimental::set_stopped_t, receiver&&) noexcept
    {
    }
    friend void tag_invoke(
        hpx::execution::experimental::set_value_t, receiver&& r, int v)
    {
        r.i = v;
    }

    int i = -1;
};

template <typename... T>
struct receiver_2
{
    friend void tag_invoke(hpx::execution::experimental::set_error_t,
        receiver_2&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_invoke(
        hpx::execution::experimental::set_stopped_t, receiver_2&&) noexcept
    {
    }
    friend void tag_invoke(
        hpx::execution::experimental::set_value_t, receiver_2&&, T...)
    {
    }
};

struct sender_1
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<int>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_stopped = false;
    };

    struct operation_state
    {
        receiver& r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(std::move(os.r), 4711);
        };
    };

    friend operation_state tag_invoke(
        hpx::execution::experimental::connect_t, sender_1&&, receiver& r)
    {
        ++friend_tag_invoke_connect_calls;
        return {r};
    }
};

struct sender_2
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<int>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_stopped = false;
    };

    struct operation_state
    {
        receiver& r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(std::move(os.r), 4711);
        };
    };
};

sender_2::operation_state tag_invoke(
    hpx::execution::experimental::connect_t, sender_2, receiver& r)
{
    ++tag_invoke_connect_calls;
    return {r};
}

struct sender_3
{
    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        sender_3 const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(int),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;

    struct operation_state
    {
        receiver& r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(std::move(os.r), 4711);
        };
    };

    friend operation_state tag_invoke(
        hpx::execution::experimental::connect_t, sender_3&&, receiver& r)
    {
        ++friend_tag_invoke_connect_calls;
        return {r};
    }
};

template <bool val, typename T>
struct sender_4
{
    struct completion_signatures
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<T>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_stopped = val;
    };
};

static std::size_t void_receiver_set_value_calls = 0;

struct void_receiver
{
    friend void tag_invoke(hpx::execution::experimental::set_error_t,
        void_receiver&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_invoke(
        hpx::execution::experimental::set_stopped_t, void_receiver&&) noexcept
    {
    }
    friend void tag_invoke(
        hpx::execution::experimental::set_value_t, void_receiver&&)
    {
        ++void_receiver_set_value_calls;
    }
};

int main()
{
    using hpx::execution::experimental::detail::has_sender_types_v;

    static_assert(!has_sender_types_v<void>,
        "void should not have completion_signatures");
    static_assert(!has_sender_types_v<std::nullptr_t>,
        "std::nullptr_t should not have completion_signatures");
    static_assert(!has_sender_types_v<int>,
        "non_sender_1 should not have completion_signatures");
    static_assert(!has_sender_types_v<double>,
        "non_sender_1 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_1>,
        "non_sender_1 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_2>,
        "non_sender_2 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_3>,
        "non_sender_3 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_4>,
        "non_sender_4 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_5>,
        "non_sender_5 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_6>,
        "non_sender_6 should not have completion_signatures");
    static_assert(!has_sender_types_v<non_sender_7>,
        "non_sender_7 should not have completion_signatures");

    using hpx::execution::experimental::is_sender_to_v;
    using hpx::execution::experimental::is_sender_v;

    static_assert(!is_sender_v<void>, "void is not a sender");
    static_assert(
        !is_sender_v<std::nullptr_t>, "std::nullptr_t is not a sender");
    static_assert(!is_sender_v<int>, "int is not a sender");
    static_assert(!is_sender_v<double>, "double is not a sender");
    static_assert(!is_sender_v<non_sender_1>, "non_sender_1 is not a sender");
    static_assert(!is_sender_v<non_sender_2>, "non_sender_2 is not a sender");
    static_assert(!is_sender_v<non_sender_3>, "non_sender_3 is not a sender");
    static_assert(!is_sender_v<non_sender_4>, "non_sender_4 is not a sender");
    static_assert(!is_sender_v<non_sender_5>, "non_sender_5 is not a sender");
    static_assert(!is_sender_v<non_sender_6>, "non_sender_6 is not a sender");
    static_assert(!is_sender_v<non_sender_7>, "non_sender_7 is not a sender");
    static_assert(is_sender_v<sender_1>, "sender_1 is a sender");
    static_assert(is_sender_v<sender_2>, "sender_2 is a sender");
    static_assert(is_sender_v<sender_3>, "sender_3 is a sender");
    static_assert(is_sender_v<sender_4<true, int>>, "sender_4 is a sender");
    static_assert(is_sender_v<sender_4<false, int>>, "sender_4 is a sender");

    static_assert(
        is_sender_to_v<sender_1, receiver>, "sender_1 is a sender to receiver");
    static_assert(!is_sender_to_v<sender_1, non_sender_1>,
        "sender_1 is not a sender to non_sender_1");
    static_assert(!is_sender_to_v<sender_1, sender_1>,
        "sender_1 is not a sender to sender_1");
    static_assert(
        is_sender_to_v<sender_2, receiver>, "sender_2 is a sender to receiver");
    static_assert(!is_sender_to_v<sender_2, non_sender_2>,
        "sender_2 is not a sender to non_sender_2");
    static_assert(!is_sender_to_v<sender_2, sender_2>,
        "sender_2 is not a sender to sender_2");

    static_assert(
        hpx::execution::experimental::is_receiver_of_v<receiver_2<int>,
            hpx::execution::experimental::completion_signatures_of_t<
                sender_4<true, int>,
                hpx::execution::experimental::env_of_t<receiver_2<int>>>>,
        "receiver_2<int> supports completion signatures of  "
        "sender_4<true,int>");
    static_assert(hpx::execution::experimental::is_receiver_of_v<receiver,
                      hpx::execution::experimental::completion_signatures_of_t<
                          sender_4<true, int>,
                          hpx::execution::experimental::env_of_t<receiver>>>,
        "receiver supports completion signatures of sender_4<true,int>");
    static_assert(
        !hpx::execution::experimental::is_receiver_of_v<receiver_2<std::string>,
            hpx::execution::experimental::completion_signatures_of_t<
                sender_4<true, int>,
                hpx::execution::experimental::env_of_t<
                    receiver_2<std::string>>>>,
        "receiver_2<int>  does not support completion signatures of "
        "sender_4<true,std::string>");
    static_assert(
        !hpx::execution::experimental::is_receiver_of_v<receiver_2<int>,
            hpx::execution::experimental::completion_signatures_of_t<
                sender_4<false, int>,
                hpx::execution::experimental::env_of_t<receiver_2<int>>>>,
        "receiver_2<int>  does not support completion signatures of "
        "sender_4<false,int>");
    static_assert(!hpx::execution::experimental::is_receiver_of_v<receiver,
                      hpx::execution::experimental::completion_signatures_of_t<
                          sender_4<false, int>,
                          hpx::execution::experimental::env_of_t<receiver>>>,
        "receiver does not support completion signatures of "
        "sender_4<false,int>");
    static_assert(
        !hpx::execution::experimental::is_receiver_of_v<receiver,
            hpx::execution::experimental::completion_signatures_of_t<sender_1,
                hpx::execution::experimental::env_of_t<receiver>>>,
        "receiver does not support completion signatures of sender_1");

    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_1, receiver>,
        "sender_1 is a sender to receiver");
    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_2, receiver>,
        "sender_2 is a sender to receiver");
    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_3, receiver>,
        "sender_3 is a sender to receiver");
    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_4<true, int>,
            receiver>,
        "sender_4<true,int> is a sender to receiver");
    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_4<false, int>,
            receiver>,
        "sender_4<false,int> is a sender to receiver");
    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_1, receiver_2<int>>,
        "sender_1 is a sender to receiver_2<int>");
    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_2, receiver_2<int>>,
        "sender_2 is a sender to receiver_2<int>");
    static_assert(
        hpx::execution::experimental::is_sender_of_v<sender_3, receiver_2<int>>,
        "sender_3 is a sender to receiver_2<int>");
    static_assert(hpx::execution::experimental::is_sender_of_v<
                      sender_4<true, std::string>, receiver_2<std::string>>,
        "sender_4<true,std::string> is a sender to receiver_2<std::string>");
    static_assert(hpx::execution::experimental::is_sender_of_v<
                      sender_4<true, std::string>, receiver_2<std::string>>,
        "sender_4<false,std::string> is a sender to receiver_2<std::string>");
    static_assert(!hpx::execution::experimental::is_sender_of_v<sender_1,
                      receiver_2<std::string>>,
        "sender_1 is not a sender to receiver_2<std::string>");
    static_assert(!hpx::execution::experimental::is_sender_of_v<sender_2,
                      receiver_2<std::string>>,
        "sender_2 is not a sender to receiver_2<std::string>");
    static_assert(!hpx::execution::experimental::is_sender_of_v<sender_3,
                      receiver_2<std::string>>,
        "sender_3 is not a sender to receiver_2<std::string>");

    {
        receiver r1;
        auto os = hpx::execution::experimental::connect(sender_1{}, r1);
        hpx::execution::experimental::start(os);
        HPX_TEST_EQ(r1.i, 4711);
        HPX_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(1));
        HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(0));
    }

    {
        receiver r2;
        auto os = hpx::execution::experimental::connect(sender_2{}, r2);
        hpx::execution::experimental::start(os);
        HPX_TEST_EQ(r2.i, 4711);
        HPX_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(1));
        HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(1));
    }

    {
        receiver r3;
        auto os = hpx::execution::experimental::connect(sender_3{}, r3);
        hpx::execution::experimental::start(os);
        HPX_TEST_EQ(r3.i, 4711);
        HPX_TEST_EQ(friend_tag_invoke_connect_calls, std::size_t(2));
        HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(1));
    }

    return hpx::util::report_errors();
}
