//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

static std::size_t start_calls = 0;
static std::size_t connect_calls = 0;
static std::size_t member_submit_calls = 0;
static std::size_t tag_invoke_submit_calls = 0;

struct receiver_1
{
    void set_error(std::exception_ptr) noexcept {}
    void set_done() noexcept {}
    void set_value(int v) noexcept
    {
        i = v;
        *is = v;
    }

    int i = -1;
    std::shared_ptr<int> is = std::make_shared<int>(-1);
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
        receiver_1& r;
        void start() noexcept
        {
            ++start_calls;
            std::move(r).set_value(4711);
        };
    };

    operation_state connect(receiver_1& r) && noexcept
    {
        ++connect_calls;
        return {r};
    }

    void submit(receiver_1& r)
    {
        ++member_submit_calls;
        std::move(*this).connect(r).start();
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
        receiver_1& r;
        void start() noexcept
        {
            ++start_calls;
            std::move(r).set_value(4711);
        };
    };

    operation_state connect(receiver_1& r) && noexcept
    {
        ++connect_calls;
        return {r};
    }

    void submit(receiver_1& r)
    {
        ++member_submit_calls;
        std::move(*this).connect(r).start();
    }
};

void tag_invoke(
    hpx::execution::experimental::submit_t, sender_2 s, receiver_1& r)
{
    ++tag_invoke_submit_calls;
    std::move(s).connect(r).start();
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
        receiver_1& r;
        void start() noexcept
        {
            ++start_calls;
            std::move(r).set_value(4711);
        };
    };

    operation_state connect(receiver_1& r) && noexcept
    {
        ++connect_calls;
        return {r};
    }
};

void tag_invoke(
    hpx::execution::experimental::submit_t, sender_3 s, receiver_1& r)
{
    ++tag_invoke_submit_calls;
    std::move(s).connect(r).start();
}

struct sender_4
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<int>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        R r;
        void start() & noexcept
        {
            ++start_calls;
            std::move(r).set_value(4711);
        };
    };

    template <typename R>
        operation_state<R> connect(R&& r) && noexcept
    {
        ++connect_calls;
        return {r};
    }
};

int main()
{
    receiver_1 r1;
    hpx::execution::experimental::submit(sender_1{}, r1);
    HPX_TEST_EQ(r1.i, 4711);
    HPX_TEST_EQ(*r1.is, 4711);
    HPX_TEST_EQ(start_calls, std::size_t(1));
    HPX_TEST_EQ(connect_calls, std::size_t(1));
    HPX_TEST_EQ(member_submit_calls, std::size_t(1));
    HPX_TEST_EQ(tag_invoke_submit_calls, std::size_t(0));

    receiver_1 r2;
    hpx::execution::experimental::submit(sender_2{}, r2);
    HPX_TEST_EQ(r2.i, 4711);
    HPX_TEST_EQ(*r2.is, 4711);
    HPX_TEST_EQ(start_calls, std::size_t(2));
    HPX_TEST_EQ(connect_calls, std::size_t(2));
    HPX_TEST_EQ(member_submit_calls, std::size_t(2));
    HPX_TEST_EQ(tag_invoke_submit_calls, std::size_t(0));

    receiver_1 r3;
    hpx::execution::experimental::submit(sender_3{}, r3);
    HPX_TEST_EQ(r3.i, 4711);
    HPX_TEST_EQ(*r3.is, 4711);
    HPX_TEST_EQ(start_calls, std::size_t(3));
    HPX_TEST_EQ(connect_calls, std::size_t(3));
    HPX_TEST_EQ(member_submit_calls, std::size_t(2));
    HPX_TEST_EQ(tag_invoke_submit_calls, std::size_t(1));

    receiver_1 r4;
    hpx::execution::experimental::submit(sender_4{}, r4);
    HPX_TEST_EQ(r4.i, -1);    // The fallback implementation copies the receiver
    HPX_TEST_EQ(*r4.is, 4711);
    HPX_TEST_EQ(start_calls, std::size_t(4));
    HPX_TEST_EQ(connect_calls, std::size_t(4));
    HPX_TEST_EQ(member_submit_calls, std::size_t(2));
    HPX_TEST_EQ(tag_invoke_submit_calls, std::size_t(1));

    return hpx::util::report_errors();
}
