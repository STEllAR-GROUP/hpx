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

static std::size_t member_connect_calls = 0;
static std::size_t tag_invoke_connect_calls = 0;

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
        void start() noexcept
        {
            std::move(r).set_value(4711);
        };
    };

    operation_state connect(receiver& r) &&
    {
        ++member_connect_calls;
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
        void start() noexcept
        {
            std::move(r).set_value(4711);
        };
    };

    operation_state connect(receiver& r) &&
    {
        ++member_connect_calls;
        return {r};
    }
};

sender_2::operation_state tag_invoke(
    hpx::execution::experimental::connect_t, sender_2, receiver& r)
{
    ++tag_invoke_connect_calls;
    return {r};
}

struct sender_3
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
        void start() noexcept
        {
            std::move(r).set_value(4711);
        };
    };
};

sender_3::operation_state tag_invoke(
    hpx::execution::experimental::connect_t, sender_3, receiver& r)
{
    ++tag_invoke_connect_calls;
    return {r};
}

static std::size_t void_receiver_set_value_calls = 0;

struct void_receiver
{
    void set_error(std::exception_ptr) noexcept {}
    void set_done() noexcept {}
    void set_value() &&
    {
        ++void_receiver_set_value_calls;
    }
};

static std::size_t member_execute_calls = 0;
static std::size_t tag_invoke_execute_calls = 0;

struct executor_1
{
    template <typename F>
    void execute(F&& f) noexcept
    {
        ++member_execute_calls;
        hpx::util::invoke(f);
    }

    bool operator==(executor_1 const&) const noexcept
    {
        return true;
    }

    bool operator!=(executor_1 const&) const noexcept
    {
        return false;
    }
};

struct executor_2
{
    template <typename F>
    void execute(F&& f) noexcept
    {
        ++member_execute_calls;
        hpx::util::invoke(f);
    }

    bool operator==(executor_2 const&) const noexcept
    {
        return true;
    }

    bool operator!=(executor_2 const&) const noexcept
    {
        return false;
    }
};

template <typename F>
void tag_invoke(hpx::execution::experimental::execute_t, executor_2, F&& f)
{
    ++tag_invoke_execute_calls;
    hpx::util::invoke(f);
}

struct executor_3
{
    bool operator==(executor_3 const&) const noexcept
    {
        return true;
    }

    bool operator!=(executor_3 const&) const noexcept
    {
        return false;
    }
};

template <typename F>
void tag_invoke(hpx::execution::experimental::execute_t, executor_3, F&& f)
{
    ++tag_invoke_execute_calls;
    hpx::util::invoke(f);
}

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
    static_assert(!unspecialized<sender_3>(nullptr),
        "sender_3 should have sender_traits");

    using hpx::execution::experimental::is_sender;
    using hpx::execution::experimental::is_sender_to;

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
    static_assert(is_sender<sender_3>::value, "sender_3 is a sender");

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
    static_assert(is_sender_to<sender_3, receiver>::value,
        "sender_3 is a sender to receiver");
    static_assert(!is_sender_to<sender_3, non_sender_3>::value,
        "sender_3 is not a sender to non_sender_3");
    static_assert(!is_sender_to<sender_3, sender_3>::value,
        "sender_3 is not a sender to sender_3");

    receiver r1;
    hpx::execution::experimental::start(
        hpx::execution::experimental::connect(sender_1{}, r1));
    HPX_TEST_EQ(r1.i, 4711);
    HPX_TEST_EQ(member_connect_calls, std::size_t(1));
    HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(0));

    receiver r2;
    hpx::execution::experimental::start(
        hpx::execution::experimental::connect(sender_2{}, r2));
    HPX_TEST_EQ(r2.i, 4711);
    HPX_TEST_EQ(member_connect_calls, std::size_t(2));
    HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(0));

    receiver r3;
    hpx::execution::experimental::start(
        hpx::execution::experimental::connect(sender_3{}, r3));
    HPX_TEST_EQ(r3.i, 4711);
    HPX_TEST_EQ(member_connect_calls, std::size_t(2));
    HPX_TEST_EQ(tag_invoke_connect_calls, std::size_t(1));

    void_receiver vr1;
    executor_1 ex1;
    hpx::execution::experimental::start(
        hpx::execution::experimental::connect(ex1, vr1));
    HPX_TEST_EQ(void_receiver_set_value_calls, std::size_t(1));
    HPX_TEST_EQ(member_execute_calls, std::size_t(1));
    HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(0));

    void_receiver vr2;
    executor_2 ex2;
    hpx::execution::experimental::start(
        hpx::execution::experimental::connect(ex2, vr2));
    HPX_TEST_EQ(void_receiver_set_value_calls, std::size_t(2));
    HPX_TEST_EQ(member_execute_calls, std::size_t(2));
    HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(0));

    void_receiver vr3;
    executor_3 ex3;
    hpx::execution::experimental::start(
        hpx::execution::experimental::connect(ex3, vr3));
    HPX_TEST_EQ(void_receiver_set_value_calls, std::size_t(3));
    HPX_TEST_EQ(member_execute_calls, std::size_t(2));
    HPX_TEST_EQ(tag_invoke_execute_calls, std::size_t(1));

    return hpx::util::report_errors();
}
