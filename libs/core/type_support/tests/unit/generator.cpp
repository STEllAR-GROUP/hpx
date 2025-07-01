//  Copyright (c) 2023-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// see: Casey Carter, Lewis Baker, Corentin Jabot. std::generator
// implementation. https://godbolt.org/z/5hcaPcfvP

#include <hpx/config.hpp>

// clang up to V12 (Apple clang up to v15) and gcc above V13 refuse to compile
// the code below
#if defined(HPX_HAVE_CXX20_COROUTINES) &&                                      \
    (!defined(HPX_CLANG_VERSION) || HPX_CLANG_VERSION >= 130000) &&            \
    (!defined(HPX_GCC_VERSION) || HPX_GCC_VERSION < 140000) &&                 \
    (!defined(HPX_APPLE_CLANG_VERSION) || HPX_APPLE_CLANG_VERSION >= 160000)

#include <hpx/generator.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace tests {

    // Simple non-nested serial generator
    hpx::generator<std::uint64_t> fib(int max)
    {
        auto a = 0, b = 1;
        for (auto n = 0; n < max; ++n)
        {
            co_yield std::exchange(a, std::exchange(b, a + b));
        }
    }

    hpx::generator<int> other_generator(int i, int j)
    {
        while (i != j)
        {
            // gcc reports -Werrer,-Wunsequenced without retval
            int retval = i++;
            co_yield retval;
        }
    }

    // Following examples show difference between:
    //
    //                                    If I co_yield a...
    //                              X / X&&  | X&         | const X&
    //                           ------------+------------+-----------
    // - generator<X, X>               (same as generator<X, X&&>)
    // - generator<X, const X&>   ref        | ref        | ref
    // - generator<X, X&&>        ref        | ill-formed | ill-formed
    // - generator<X, X&>         ill-formed | ref        | ill-formed
    struct X
    {
        int id;

        explicit X(int id)
          : id(id)
        {
        }

        X(X const& x)
          : id(x.id)
        {
        }
        X(X&& x)
          : id(std::exchange(x.id, -1))
        {
        }
        ~X() {}

        bool operator==(X const& rhs) const noexcept
        {
            return id == rhs.id;
        }

        friend std::ostream& operator<<(std::ostream& os, X const& rhs) noexcept
        {
            os << rhs.id;
            return os;
        }
    };

    hpx::generator<X> always_ref_example()
    {
        co_yield X{1};
        {
            X x{2};
            co_yield x;
            HPX_TEST(x.id == 2);
        }
        {
            X const x{3};
            co_yield x;
            HPX_TEST(x.id == 3);
        }
        {
            X x{4};
            co_yield std::move(x);
        }
    }

    hpx::generator<X&&> xvalue_example()
    {
        co_yield X{1};
        X x{2};
        co_yield x;    // well-formed: generated element is copy of lvalue
        HPX_TEST(x.id == 2);
        co_yield std::move(x);
    }

    hpx::generator<X const&> const_lvalue_example()
    {
        co_yield X{1};    // OK
        X const x{2};

        co_yield x;               // OK
        co_yield std::move(x);    // OK: same as above
    }

    hpx::generator<X&> lvalue_example()
    {
        // co_yield X{1}; // ill-formed: prvalue -> non-const lvalue
        X x{2};
        co_yield x;    // OK
        // co_yield std::move(x); // ill-formed: xvalue -> non-const lvalue
    }

    // These examples show different usages of reference/value_type
    // template parameters

    // value_type = std::unique_ptr<int>
    // reference = std::unique_ptr<int>&&
    hpx::generator<std::unique_ptr<int>&&> unique_ints(int const high)
    {
        for (auto i = 0; i < high; ++i)
        {
            co_yield std::make_unique<int>(i);
        }
    }

    // value_type = std::string_view
    // reference = std::string_view&&
    hpx::generator<std::string_view> string_views()
    {
        co_yield "foo";
        co_yield "bar";
    }

    // value_type = std::string
    // reference = std::string_view
    template <typename Allocator>
    hpx::generator<std::string_view, std::string> strings(
        std::allocator_arg_t, Allocator)
    {
        co_yield {};
        co_yield "start";
        for (auto sv : string_views())
        {
            co_yield std::string{sv} + '!';
        }
        co_yield "end";
    }

    template <typename T>
    struct stateful_allocator
    {
        using value_type = T;

        int id;

        explicit stateful_allocator(int id) noexcept
          : id(id)
        {
        }

        template <typename U>
        stateful_allocator(stateful_allocator<U> const& x)
          : id(x.id)
        {
        }

        T* allocate(std::size_t count)
        {
            return std::allocator<T>().allocate(count);
        }

        void deallocate(T* ptr, std::size_t count) noexcept
        {
            std::allocator<T>().deallocate(ptr, count);
        }

        template <typename U>
        bool operator==(stateful_allocator<U> const& x) const
        {
            return this->id == x.id;
        }
    };

    // gcc V11 and on are complaining about mismatched-new-delete
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 150000
    hpx::generator<int, void, std::allocator<std::byte>> stateless_example()
    {
        co_yield 42;
    }

    hpx::generator<int, void, std::allocator<std::byte>> stateless_example(
        std::allocator_arg_t, std::allocator<std::byte>)
    {
        co_yield 42;
    }
#endif

    template <typename Allocator>
    hpx::generator<int, void, Allocator> stateful_alloc_example(
        std::allocator_arg_t, Allocator)
    {
        co_yield 42;
    }

    struct member_coro
    {
        hpx::generator<int> f() const
        {
            co_yield 42;
        }
    };
}    // namespace tests

int main()
{
    {
        std::vector const expected = {
            0ULL, 1ULL, 1ULL, 2ULL, 3ULL, 5ULL, 8ULL, 13ULL, 21ULL, 34ULL};
        std::size_t i = 0;
        for (auto&& x : tests::fib(10))
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    {
        std::vector const expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::size_t i = 0;
        for (auto&& x : tests::other_generator(1, 10))
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    {
        std::vector const expected = {
            tests::X(1), tests::X(2), tests::X(3), tests::X(4)};
        std::size_t i = 0;
        for (auto&& x : tests::always_ref_example())
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    {
        std::vector const expected = {tests::X(1), tests::X(2), tests::X(2)};
        std::size_t i = 0;
        for (auto&& x : tests::xvalue_example())
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    {
        std::vector const expected = {tests::X(1), tests::X(2), tests::X(2)};
        std::size_t i = 0;
        for (auto&& x : tests::const_lvalue_example())
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    {
        std::vector const expected = {tests::X(2)};
        std::size_t i = 0;
        for (auto&& x : tests::lvalue_example())
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    {
        std::vector const expected = {0, 1, 2, 3, 4};
        std::size_t i = 0;
        for (std::unique_ptr<int> ptr : tests::unique_ints(5))
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(*ptr, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    // gcc V11 and on are complaining about mismatched-new-delete
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 150000
    {
        std::vector const expected = {42};
        std::size_t i = 0;
        for (auto&& x : tests::stateless_example())
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }

    {
        std::vector const expected = {42};
        std::size_t i = 0;
        for (auto&& x : tests::stateless_example(
                 std::allocator_arg, std::allocator<float>{}))
        {
            HPX_TEST_LT(i, expected.size());
            HPX_TEST_EQ(x, expected[i++]);
        }
        HPX_TEST_EQ(i, expected.size());
    }
#endif

    constexpr tests::member_coro m;
    HPX_TEST(*m.f().begin() == 42);

    return hpx::util::report_errors();
}

#else
int main()
{
    return 0;
}
#endif
