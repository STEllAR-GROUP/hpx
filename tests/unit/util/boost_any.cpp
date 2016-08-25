// Copyright Kevlin Henney, 2000, 2001. All rights reserved.
// Copyright (c) 2013 Hartmut Kaiser.
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/util/any.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "small_big_object.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace any_tests // test suite
{
    void test_default_ctor();
    void test_converting_ctor();
    void test_copy_ctor();
    void test_copy_assign();
    void test_converting_assign();
    void test_bad_cast();
    void test_swap_small();
    void test_swap_big();
    void test_null_copying();
    void test_cast_to_reference();

    struct test_case
    {
        char const* const name;
        void (*test_func)();
    };

    const test_case test_cases[] =
    {
        { "default construction",           test_default_ctor      },
        { "single argument construction",   test_converting_ctor   },
        { "copy construction",              test_copy_ctor         },
        { "copy assignment operator",       test_copy_assign       },
        { "converting assignment operator", test_converting_assign },
        { "failed custom keyword cast",     test_bad_cast          },
        { "swap member function, small",    test_swap_small        },
        { "swap member function, big",      test_swap_big          },
        { "copying operations on a null",   test_null_copying      },
        { "cast to reference types",        test_cast_to_reference }
    };

    typedef test_case const* test_case_iterator;

    test_case_iterator begin_tests = test_cases;
    const test_case_iterator end_tests =
        test_cases + (sizeof test_cases / sizeof *test_cases);

    struct copy_counter
    {
    public:

        copy_counter() {}
        copy_counter(const copy_counter&) { ++count; }
        copy_counter& operator=(const copy_counter&) { ++count; return *this; }
        static int get_count() { return count; }

    private:
        static int count;
    };

    bool operator==(copy_counter const& lhs, copy_counter const& rhs)
    {
        return true;
    }

    int copy_counter::count = 0;
}

namespace std
{
    std::ostream& operator<<(std::ostream& os, boost::detail::sp_typeinfo const& ti)
    {
        return os;
    }

    std::ostream& operator<<(std::ostream& os, any_tests::copy_counter const& cc)
    {
        return os;
    }

    std::istream& operator>>(std::istream& is, any_tests::copy_counter& cc)
    {
        return is;
    }
}

namespace any_tests // test definitions
{
    using hpx::util::any;
    using hpx::util::any_cast;

    void test_default_ctor()
    {
        const any value;

        HPX_TEST_MSG(value.empty(), "empty");
        HPX_TEST_EQ_MSG(static_cast<void*>(nullptr),
            any_cast<int>(&value), "any_cast<int>");
        HPX_TEST_EQ_MSG(value.type(), BOOST_SP_TYPEID(hpx::util::detail::any::empty),
            "type");
    }

    void test_converting_ctor()
    {
        std::string text = "test message";
        any value = any(text);

        HPX_TEST_EQ_MSG(false, value.empty(), "empty");
        HPX_TEST_EQ_MSG(value.type(), typeid(std::string), "type");
        HPX_TEST_EQ_MSG(static_cast<void*>(nullptr),
            any_cast<int>(&value), "any_cast<int>");
        HPX_TEST_NEQ_MSG(static_cast<void*>(nullptr), any_cast<std::string>(&value),
            "any_cast<std::string>");
        HPX_TEST_EQ_MSG(
            any_cast<std::string>(value), text,
            "comparing cast copy against original text");
        HPX_TEST_NEQ_MSG(
            any_cast<std::string>(&value), &text,
            "comparing address in copy against original text");
    }

    void test_copy_ctor()
    {
        std::string text = "test message";
        any original = any(text), copy = any(original);

        HPX_TEST_EQ_MSG(false, copy.empty(), "empty");
        HPX_TEST_EQ_MSG(original.type(), copy.type(), "type");
        HPX_TEST_EQ_MSG(
            any_cast<std::string>(original), any_cast<std::string>(copy),
            "comparing cast copy against original");
        HPX_TEST_EQ_MSG(
            text, any_cast<std::string>(copy),
            "comparing cast copy against original text");
        HPX_TEST_NEQ_MSG(
            any_cast<std::string>(&original),
            any_cast<std::string>(&copy),
            "comparing address in copy against original");
    }

    void test_copy_assign()
    {
        std::string text = "test message";
        any original = any(text), copy;
        any * assign_result = &(copy = original);

        HPX_TEST_EQ_MSG(false, copy.empty(), "empty");
        HPX_TEST_EQ_MSG(original.type(), copy.type(), "type");
        HPX_TEST_EQ_MSG(
            any_cast<std::string>(original), any_cast<std::string>(copy),
            "comparing cast copy against cast original");
        HPX_TEST_EQ_MSG(
            text, any_cast<std::string>(copy),
            "comparing cast copy against original text");
        HPX_TEST_NEQ_MSG(
            any_cast<std::string>(&original),
            any_cast<std::string>(&copy),
            "comparing address in copy against original");
        HPX_TEST_EQ_MSG(assign_result, &copy, "address of assignment result");
    }

    void test_converting_assign()
    {
        std::string text = "test message";
        any value;
        any * assign_result = &(value = text);

        HPX_TEST_EQ_MSG(false, value.empty(), "type");
        HPX_TEST_EQ_MSG(value.type(), typeid(std::string), "type");
        HPX_TEST_EQ_MSG(static_cast<void*>(nullptr),
            any_cast<int>(&value), "any_cast<int>");
        HPX_TEST_NEQ_MSG(static_cast<void*>(nullptr), any_cast<std::string>(&value),
            "any_cast<std::string>");
        HPX_TEST_EQ_MSG(
            any_cast<std::string>(value), text,
            "comparing cast copy against original text");
        HPX_TEST_NEQ_MSG(
            any_cast<std::string>(&value),
            &text,
            "comparing address in copy against original text");
        HPX_TEST_EQ_MSG(assign_result, &value, "address of assignment result");
    }

    void test_bad_cast()
    {
        std::string text = "test message";
        any value = any(text);

        {
            bool caught_exception = false;
            try {
                any_cast<const char *>(value);
            }
            catch(hpx::util::bad_any_cast const&) {
                caught_exception = true;
            }
            catch(...) {
                HPX_TEST_MSG(false, "caught wrong exception");
            }
            HPX_TEST(caught_exception);
        }
    }

    void test_swap_small()
    {
        if (sizeof(small_object) <= sizeof(void*))
            std::cout << "object is small\n";
        else
            std::cout << "object is large\n";

        small_object text = 17;
        any original = any(text), swapped;
        small_object * original_ptr = any_cast<small_object>(&original);
        any * swap_result = &original.swap(swapped);

        HPX_TEST_MSG(original.empty(), "empty on original");
        HPX_TEST_EQ_MSG(false, swapped.empty(), "empty on swapped");
        HPX_TEST_EQ_MSG(swapped.type(), typeid(small_object), "type");
        HPX_TEST_EQ_MSG(
            text, any_cast<small_object>(swapped),
            "comparing swapped copy against original text");
        HPX_TEST_NEQ_MSG(static_cast<void*>(nullptr),
            original_ptr, "address in pre-swapped original");
        HPX_TEST_NEQ_MSG(
            original_ptr,
            any_cast<small_object>(&swapped),
            "comparing address in swapped against original");
        HPX_TEST_EQ_MSG(swap_result, &original, "address of swap result");

        any copy1 = any(copy_counter());
        any copy2 = any(copy_counter());
        int count = copy_counter::get_count();
        swap(copy1, copy2);
        HPX_TEST_EQ_MSG(count, copy_counter::get_count(),
            "checking that free swap doesn't make any copies.");
    }

    void test_swap_big()
    {
        if (sizeof(big_object) <= sizeof(void*))
            std::cout << "object is small\n";
        else
            std::cout << "object is large\n";

        big_object text(5, 12);
        any original = any(text), swapped;
        big_object * original_ptr = any_cast<big_object>(&original);
        any * swap_result = &original.swap(swapped);

        HPX_TEST_MSG(original.empty(), "empty on original");
        HPX_TEST_EQ_MSG(false, swapped.empty(), "empty on swapped");
        HPX_TEST_EQ_MSG(swapped.type(), typeid(big_object), "type");
        HPX_TEST_EQ_MSG(
            text, any_cast<big_object>(swapped),
            "comparing swapped copy against original text");
        HPX_TEST_NEQ_MSG(static_cast<void*>(nullptr),
            original_ptr, "address in pre-swapped original");
        HPX_TEST_EQ_MSG(
            original_ptr,
            any_cast<big_object>(&swapped),
            "comparing address in swapped against original");
        HPX_TEST_EQ_MSG(swap_result, &original, "address of swap result");

        any copy1 = any(copy_counter());
        any copy2 = any(copy_counter());
        int count = copy_counter::get_count();
        swap(copy1, copy2);
        HPX_TEST_EQ_MSG(count, copy_counter::get_count(),
            "checking that free swap doesn't make any copies.");
    }

    void test_null_copying()
    {
        const any null;
        any copied = null, assigned;
        assigned = null;

        HPX_TEST_MSG(null.empty(), "empty on null");
        HPX_TEST_MSG(copied.empty(), "empty on copied");
        HPX_TEST_MSG(assigned.empty(), "empty on copied");
    }

    void test_cast_to_reference()
    {
        any a(137);
        const any b(a);

        int &                ra    = any_cast<int &>(a);
        int const &          ra_c  = any_cast<int const &>(a);
        int volatile &       ra_v  = any_cast<int volatile &>(a);
        int const volatile & ra_cv = any_cast<int const volatile&>(a);

        HPX_TEST_MSG(
            &ra == &ra_c && &ra == &ra_v && &ra == &ra_cv,
            "cv references to same obj");

        int const &          rb_c  = any_cast<int const &>(b);
        int const volatile & rb_cv = any_cast<int const volatile &>(b);

        HPX_TEST_MSG(&rb_c == &rb_cv, "cv references to copied const obj");
        HPX_TEST_MSG(&ra != &rb_c, "copies hold different objects");

        ++ra;
        int incremented = any_cast<int>(a);
        HPX_TEST_MSG(incremented == 138, "increment by reference changes value");

        {
            bool caught_exception = false;
            try {
                any_cast<char &>(a);
            }
            catch(hpx::util::bad_any_cast const&) {
                caught_exception = true;
            }
            catch(...) {
                HPX_TEST_MSG(false, "caught wrong exception");
            }
            HPX_TEST(caught_exception);
        }

        {
            bool caught_exception = false;
            try {
                any_cast<const char &>(b);
            }
            catch(hpx::util::bad_any_cast const&) {
                caught_exception = true;
            }
            catch(...) {
                HPX_TEST_MSG(false, "caught wrong exception");
            }
            HPX_TEST(caught_exception);
        }
    }
}

int main()
{
    using namespace any_tests;
    while (begin_tests != end_tests)
    {
        (*begin_tests->test_func)();
        ++begin_tests;
    }
    return hpx::util::report_errors();
}


