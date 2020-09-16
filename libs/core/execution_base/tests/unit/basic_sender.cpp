//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

bool connect_called = false;

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
    void set_error(std::exception_ptr) noexcept {}
    void set_done() noexcept {}
    void set_value(int v) &&
    {
        i = v;
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
        void start() && noexcept
        {
            std::move(r).set_value(4711);
        };
    };

    operation_state connect(receiver& r) &&
    {
        return {r};
    }
};

template <typename Sender>
constexpr bool unspecialized(...)
{
    return false;
}

template <typename Sender>
constexpr bool unspecialized(
    typename hpx::execution_base::experimental::traits::sender_traits<
        Sender>::__unspecialized*)
{
    return true;
}

int main()
{
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

    using hpx::execution_base::experimental::traits::is_sender;
    using hpx::execution_base::experimental::traits::is_sender_to;

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

    static_assert(is_sender_to<sender_1, receiver>::value,
        "sender_1 is a sender to receiver");
    static_assert(!is_sender_to<sender_1, non_sender_1>::value,
        "sender_1 is not a sender to non_sender_1");
    static_assert(!is_sender_to<sender_1, sender_1>::value,
        "sender_1 is not a sender to sender_1");

    receiver r;
    hpx::execution_base::experimental::start(
        hpx::execution_base::experimental::connect(sender_1{}, r));

    HPX_TEST_EQ(r.i, 4711);

    return hpx::util::report_errors();
}
