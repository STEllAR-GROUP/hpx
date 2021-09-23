//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

static std::size_t friend_tag_dispatch_connect_calls = 0;
static std::size_t tag_dispatch_connect_calls = 0;

struct non_sender_1
{
};

struct non_sender_2
{
    template <template <class...> class Variant>
    using error_types = Variant<>;

    static constexpr bool sends_done = false;
};

struct non_sender_3
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    static constexpr bool sends_done = false;
};

struct non_sender_4
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<>;
};

struct non_sender_5
{
    static constexpr bool sends_done = false;
};

struct non_sender_6
{
    template <template <class...> class Variant>
    using error_types = Variant<>;
};

struct non_sender_7
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;
};

struct receiver
{
    friend void tag_dispatch(hpx::execution::experimental::set_error_t,
        receiver&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_dispatch(
        hpx::execution::experimental::set_done_t, receiver&&) noexcept
    {
    }
    friend void tag_dispatch(
        hpx::execution::experimental::set_value_t, receiver&& r, int v)
    {
        r.i = v;
    }

    int i = -1;
};

struct sender_1
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<int>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    struct operation_state
    {
        receiver& r;
        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(std::move(os.r), 4711);
        };
    };

    friend operation_state tag_dispatch(
        hpx::execution::experimental::connect_t, sender_1&&, receiver& r)
    {
        ++friend_tag_dispatch_connect_calls;
        return {r};
    }
};

struct sender_2
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<int>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    struct operation_state
    {
        receiver& r;
        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(std::move(os.r), 4711);
        };
    };
};

sender_2::operation_state tag_dispatch(
    hpx::execution::experimental::connect_t, sender_2, receiver& r)
{
    ++tag_dispatch_connect_calls;
    return {r};
}

static std::size_t void_receiver_set_value_calls = 0;

struct void_receiver
{
    friend void tag_dispatch(hpx::execution::experimental::set_error_t,
        void_receiver&&, std::exception_ptr) noexcept
    {
    }
    friend void tag_dispatch(
        hpx::execution::experimental::set_done_t, void_receiver&&) noexcept
    {
    }
    friend void tag_dispatch(
        hpx::execution::experimental::set_value_t, void_receiver&&)
    {
        ++void_receiver_set_value_calls;
    }
};

template <typename Sender>
constexpr bool unspecialized(...)
{
    return false;
}

template <typename Sender>
constexpr bool unspecialized(
    typename hpx::execution::experimental::sender_traits<
        Sender>::__unspecialized*)
{
    return true;
}

int main()
{
    static_assert(
        unspecialized<void>(nullptr), "void should not have sender_traits");
    static_assert(unspecialized<std::nullptr_t>(nullptr),
        "std::nullptr_t should not have sender_traits");
    static_assert(unspecialized<int>(nullptr),
        "non_sender_1 should not have sender_traits");
    static_assert(unspecialized<double>(nullptr),
        "non_sender_1 should not have sender_traits");
    static_assert(unspecialized<non_sender_1>(nullptr),
        "non_sender_1 should not have sender_traits");
    static_assert(unspecialized<non_sender_2>(nullptr),
        "non_sender_2 should not have sender_traits");
    static_assert(unspecialized<non_sender_3>(nullptr),
        "non_sender_3 should not have sender_traits");
    static_assert(unspecialized<non_sender_4>(nullptr),
        "non_sender_4 should not have sender_traits");
    static_assert(unspecialized<non_sender_5>(nullptr),
        "non_sender_5 should not have sender_traits");
    static_assert(unspecialized<non_sender_6>(nullptr),
        "non_sender_6 should not have sender_traits");
    static_assert(unspecialized<non_sender_7>(nullptr),
        "non_sender_7 should not have sender_traits");
    static_assert(!unspecialized<sender_1>(nullptr),
        "sender_1 should have sender_traits");
    static_assert(!unspecialized<sender_2>(nullptr),
        "sender_2 should have sender_traits");

    using hpx::execution::experimental::is_sender;
    using hpx::execution::experimental::is_sender_to;

    static_assert(!is_sender<void>::value, "void is not a sender");
    static_assert(
        !is_sender<std::nullptr_t>::value, "std::nullptr_t is not a sender");
    static_assert(!is_sender<int>::value, "int is not a sender");
    static_assert(!is_sender<double>::value, "double is not a sender");
    static_assert(
        !is_sender<non_sender_1>::value, "non_sender_1 is not a sender");
    static_assert(
        !is_sender<non_sender_2>::value, "non_sender_2 is not a sender");
    static_assert(
        !is_sender<non_sender_3>::value, "non_sender_3 is not a sender");
    static_assert(
        !is_sender<non_sender_4>::value, "non_sender_4 is not a sender");
    static_assert(
        !is_sender<non_sender_5>::value, "non_sender_5 is not a sender");
    static_assert(
        !is_sender<non_sender_6>::value, "non_sender_6 is not a sender");
    static_assert(
        !is_sender<non_sender_7>::value, "non_sender_7 is not a sender");
    static_assert(is_sender<sender_1>::value, "sender_1 is a sender");
    static_assert(is_sender<sender_2>::value, "sender_2 is a sender");

    static_assert(is_sender_to<sender_1, receiver>::value,
        "sender_1 is a sender to receiver");
    static_assert(!is_sender_to<sender_1, non_sender_1>::value,
        "sender_1 is not a sender to non_sender_1");
    static_assert(!is_sender_to<sender_1, sender_1>::value,
        "sender_1 is not a sender to sender_1");
    static_assert(is_sender_to<sender_2, receiver>::value,
        "sender_2 is a sender to receiver");
    static_assert(!is_sender_to<sender_2, non_sender_2>::value,
        "sender_2 is not a sender to non_sender_2");
    static_assert(!is_sender_to<sender_2, sender_2>::value,
        "sender_2 is not a sender to sender_2");

    {
        receiver r1;
        auto os = hpx::execution::experimental::connect(sender_1{}, r1);
        hpx::execution::experimental::start(os);
        HPX_TEST_EQ(r1.i, 4711);
        HPX_TEST_EQ(friend_tag_dispatch_connect_calls, std::size_t(1));
        HPX_TEST_EQ(tag_dispatch_connect_calls, std::size_t(0));
    }

    {
        receiver r2;
        auto os = hpx::execution::experimental::connect(sender_2{}, r2);
        hpx::execution::experimental::start(os);
        HPX_TEST_EQ(r2.i, 4711);
        HPX_TEST_EQ(friend_tag_dispatch_connect_calls, std::size_t(1));
        HPX_TEST_EQ(tag_dispatch_connect_calls, std::size_t(1));
    }

    return hpx::util::report_errors();
}
