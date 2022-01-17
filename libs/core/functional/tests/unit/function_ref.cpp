//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2001-2003.
//  Copyright 2013 Hartmut Kaiser
//  Copyright 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

#include <hpx/functional/function_ref.hpp>
#include <hpx/modules/testing.hpp>

#include <functional>
#include <string>
#include <utility>

using std::string;

int global_int;

struct write_five_obj
{
    void operator()() const
    {
        global_int = 5;
    }
};
struct write_three_obj
{
    int operator()() const
    {
        global_int = 3;
        return 7;
    }
};
static void write_five()
{
    global_int = 5;
}
static void write_three()
{
    global_int = 3;
}
struct generate_five_obj
{
    int operator()() const
    {
        return 5;
    }
};
struct generate_three_obj
{
    int operator()() const
    {
        return 3;
    }
};
static int generate_five()
{
    return 5;
}
static int generate_three()
{
    return 3;
}
static string identity_str(const string& s)
{
    return s;
}
static string string_cat(const string& s1, const string& s2)
{
    return s1 + s2;
}
static int sum_ints(int x, int y)
{
    return x + y;
}

struct write_const_1_nonconst_2
{
    void operator()()
    {
        global_int = 2;
    }
    void operator()() const
    {
        global_int = 1;
    }
};

struct add_to_obj
{
    add_to_obj(int v)
      : value(v)
    {
    }

    int operator()(int x) const
    {
        return value + x;
    }

    int value;
};

static void test_zero_args()
{
    typedef hpx::util::function_ref<void()> func_void_type;

    write_five_obj five;
    write_three_obj three;

    // Invocation of a function
    func_void_type v1 = five;
    global_int = 0;
    v1();
    HPX_TEST_EQ(global_int, 5);

    // Invocation and self-assignment
    v1 = three;
    global_int = 0;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
    v1 = v1;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    v1();
    HPX_TEST_EQ(global_int, 3);

    // Assignment to a function
    v1 = five;

    // Invocation and self-assignment
    global_int = 0;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
    v1 = (v1);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    v1();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v1 = write_five;
    global_int = 0;
    v1();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v1 = write_three;
    global_int = 0;
    v1();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v1 = five;
    global_int = 0;
    v1();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v1 = &write_three;
    global_int = 0;
    v1();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    func_void_type v2(v1);
    v2 = three;
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v2 = (five);
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v2 = (write_five);
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v2 = write_three;
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 3);

    // Swapping
    v1 = five;
    std::swap(v1, v2);
    v2();
    HPX_TEST_EQ(global_int, 5);
    v1();
    HPX_TEST_EQ(global_int, 3);
    std::swap(v1, v2);

    // Invocation
    v2 = five;
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v2 = &write_three;
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v1 = three;
    v2 = v1;
    global_int = 0;
    v1();
    HPX_TEST_EQ(global_int, 3);
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 3);

    // Assign to a function from a function with a function
    v2 = write_five;
    v1 = v2;
    global_int = 0;
    v1();
    HPX_TEST_EQ(global_int, 5);
    global_int = 0;
    v2();
    HPX_TEST_EQ(global_int, 5);

    // Construct a function given another function containing a function
    func_void_type v3(v1);

    // Invocation of a function
    global_int = 0;
    v3();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v3 = three;
    global_int = 0;
    v3();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v3 = five;
    global_int = 0;
    v3();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v3 = &write_five;
    global_int = 0;
    v3();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v3 = &write_three;
    global_int = 0;
    v3();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v3 = five;
    global_int = 0;
    v3();
    HPX_TEST_EQ(global_int, 5);

    // Construction of a function from a function containing a functor
    func_void_type v4(v3);

    // Invocation of a function
    global_int = 0;
    v4();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v4 = three;
    global_int = 0;
    v4();
    HPX_TEST_EQ(global_int, 3);

    // Assignment to a function
    v4 = five;

    // Invocation
    global_int = 0;
    v4();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v4 = &write_five;
    global_int = 0;
    v4();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v4 = &write_three;
    global_int = 0;
    v4();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v4 = five;
    global_int = 0;
    v4();
    HPX_TEST_EQ(global_int, 5);

    // Construction of a function from a functor
    func_void_type v5(five);

    // Invocation of a function
    global_int = 0;
    v5();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v5 = three;
    global_int = 0;
    v5();
    HPX_TEST_EQ(global_int, 3);

    // Assignment to a function
    v5 = five;

    // Invocation
    global_int = 0;
    v5();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v5 = &write_five;
    global_int = 0;
    v5();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v5 = &write_three;
    global_int = 0;
    v5();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v5 = five;
    global_int = 0;
    v5();
    HPX_TEST_EQ(global_int, 5);

    // Construction of a function from a function
    func_void_type v6(&write_five);

    // Invocation of a function
    global_int = 0;
    v6();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v6 = three;
    global_int = 0;
    v6();
    HPX_TEST_EQ(global_int, 3);

    // Assignment to a function
    v6 = five;

    // Invocation
    global_int = 0;
    v6();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v6 = &write_five;
    global_int = 0;
    v6();
    HPX_TEST_EQ(global_int, 5);

    // Invocation
    v6 = &write_three;
    global_int = 0;
    v6();
    HPX_TEST_EQ(global_int, 3);

    // Invocation
    v6 = five;
    global_int = 0;
    v6();
    HPX_TEST_EQ(global_int, 5);

    // Const vs. non-const
    write_const_1_nonconst_2 one_or_two;
    const hpx::util::function_ref<void()> v7(one_or_two);
    hpx::util::function_ref<void()> v8(one_or_two);

    global_int = 0;
    v7();
    HPX_TEST_EQ(global_int, 2);

    global_int = 0;
    v8();
    HPX_TEST_EQ(global_int, 2);

    // Test return values
    typedef hpx::util::function_ref<int()> func_int_type;
    generate_five_obj gen_five;
    generate_three_obj gen_three;

    func_int_type i0(gen_five);

    HPX_TEST_EQ(i0(), 5);
    i0 = gen_three;
    HPX_TEST_EQ(i0(), 3);
    i0 = &generate_five;
    HPX_TEST_EQ(i0(), 5);
    i0 = &generate_three;
    HPX_TEST_EQ(i0(), 3);

    // Test return values with compatible types
    typedef hpx::util::function_ref<long()> func_long_type;
    func_long_type i1(gen_five);

    HPX_TEST_EQ(i1(), 5);
    i1 = gen_three;
    HPX_TEST_EQ(i1(), 3);
    i1 = &generate_five;
    HPX_TEST_EQ(i1(), 5);
    i1 = &generate_three;
    HPX_TEST_EQ(i1(), 3);
}

static void test_one_arg()
{
    std::negate<int> neg;

    hpx::util::function_ref<int(int)> f1(neg);
    HPX_TEST_EQ(f1(5), -5);

    hpx::util::function_ref<string(string)> id(&identity_str);
    HPX_TEST_EQ(id("str"), "str");

    hpx::util::function_ref<string(const char*)> id2(&identity_str);
    HPX_TEST_EQ(id2("foo"), "foo");

    add_to_obj add_to(5);
    hpx::util::function_ref<int(int)> f2(add_to);
    HPX_TEST_EQ(f2(3), 8);

    const hpx::util::function_ref<int(int)> cf2(add_to);
    HPX_TEST_EQ(cf2(3), 8);
}

static void test_two_args()
{
    hpx::util::function_ref<string(const string&, const string&)> cat(
        &string_cat);
    HPX_TEST_EQ(cat("str", "ing"), "string");

    hpx::util::function_ref<int(short, short)> sum(&sum_ints);
    HPX_TEST_EQ(sum(2, 3), 5);
}

struct add_with_throw_on_copy
{
    int operator()(int x, int y) const
    {
        return x + y;
    }

    add_with_throw_on_copy() {}

    add_with_throw_on_copy(const add_with_throw_on_copy&)
    {
        throw std::runtime_error("But this CAN'T throw");
    }

    add_with_throw_on_copy& operator=(const add_with_throw_on_copy&)
    {
        throw std::runtime_error("But this CAN'T throw");
    }
};

static void test_ref()
{
    add_with_throw_on_copy atc;
    try
    {
        hpx::util::function_ref<int(int, int)> f(std::ref(atc));
        HPX_TEST_EQ(f(1, 3), 4);
    }
    catch (std::runtime_error const& /*e*/)
    {
        HPX_TEST_MSG(false, "Nonthrowing constructor threw an exception");
    }
}

static void test_ptr_ref()
{
    typedef hpx::util::function_ref<void()> func_void_type;
    typedef hpx::util::function_ref<int()> func_int_type;

    // Invocation of a function
    void (*void_ptr)() = &write_five;
    func_void_type v1 = void_ptr;
    global_int = 0;
    void_ptr = &write_three;
    v1();
    HPX_TEST_EQ(global_int, 5);

    // Invocation and assignment
    void_ptr = &write_five;
    v1 = void_ptr;
    global_int = 0;
    void_ptr = &write_three;
    v1();
    HPX_TEST_EQ(global_int, 5);

    // Invocation of a function
    int (*int_ptr)() = &generate_five;
    func_int_type v2 = int_ptr;
    int_ptr = &generate_three;
    HPX_TEST_EQ(v2(), 5);

    // Invocation and assignment
    int_ptr = &generate_five;
    v2 = int_ptr;
    int_ptr = &generate_three;
    HPX_TEST_EQ(v2(), 5);
}

struct big_aggregating_structure
{
    // int disable_small_objects_optimizations[32];

    big_aggregating_structure()
    {
        ++global_int;
    }

    big_aggregating_structure(const big_aggregating_structure&)
    {
        ++global_int;
    }

    ~big_aggregating_structure()
    {
        --global_int;
    }

    void operator()()
    {
        ++global_int;
    }

    void operator()(int)
    {
        ++global_int;
    }
};

static void test_copy_semantics()
{
    typedef hpx::util::function_ref<void()> f1_type;

    big_aggregating_structure obj;

    f1_type f1 = obj;
    global_int = 0;
    f1();
    HPX_TEST_EQ(global_int, 1);

    // Testing rvalue constructors
    f1_type f2(static_cast<f1_type&&>(f1));
    HPX_TEST_EQ(global_int, 1);
    f2();
    HPX_TEST_EQ(global_int, 2);

    f1_type f3(static_cast<f1_type&&>(f2));
    HPX_TEST_EQ(global_int, 2);
    f3();
    HPX_TEST_EQ(global_int, 3);

    // Testing, that no copies are made
    f1_type f4 = obj;
    HPX_TEST_EQ(global_int, 3);
    f1_type f5 = obj;
    HPX_TEST_EQ(global_int, 3);
    f4 = static_cast<f1_type&&>(f5);
    HPX_TEST_EQ(global_int, 3);
}

int main(int, char*[])
{
    test_zero_args();
    test_one_arg();
    test_two_args();
    test_ref();
    test_ptr_ref();
    test_copy_semantics();

    return hpx::util::report_errors();
}
