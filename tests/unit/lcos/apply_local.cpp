//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::atomic<boost::int32_t> accumulator;

void increment(boost::int32_t i)
{
    accumulator += i;
}

///////////////////////////////////////////////////////////////////////////////
struct increment_function_object
{
    // implement result_of protocol
    template <typename F>
    struct result;

    template <typename F, typename T>
    struct result<F(T)>
    {
        typedef void type;
    };

    // actual functionality
    void operator()(boost::int32_t i) const
    {
        accumulator += i;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct increment_type
{
    void call(boost::int32_t i) const
    {
        accumulator += i;
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        using hpx::util::placeholders::_1;

        hpx::apply(&increment, 1);
        hpx::apply(hpx::util::bind(&increment, 1));
        hpx::apply(hpx::util::bind(&increment, _1), 1);
    }

    {
        using hpx::util::placeholders::_1;

        hpx::apply(increment, 1);
        hpx::apply(hpx::util::bind(increment, 1));
        hpx::apply(hpx::util::bind(increment, _1), 1);
    }

    {
        increment_type inc;

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        hpx::apply(&increment_type::call, inc, 1);
        hpx::apply(hpx::util::bind(&increment_type::call, inc, 1));
        hpx::apply(hpx::util::bind(&increment_type::call, inc, _1), 1);
    }

    {
        increment_function_object obj;

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

// We are currently not able to detect whether a type is callable. We need the
// C++11 is_callable trait for this. For now, please use hpx::util::bind to
// wrap your function object in order to pass it to async (see below).
//
//         hpx::apply(obj, 1);

        hpx::apply(hpx::util::bind(obj, 1));
        hpx::apply(hpx::util::bind(obj, _1), 1);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    accumulator.store(0);

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    HPX_TEST_EQ(accumulator.load(), 11);

    return hpx::util::report_errors();
}

