//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
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

#if !defined(BOOST_NO_CXX11_LAMBDAS) || !defined( BOOST_NO_CXX11_AUTO_DECLARATIONS)
auto increment_lambda = [](boost::int32_t i){ accumulator += i; };
#endif /*BOOST_NO_CXX11_LAMBDAS*/

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

        hpx::apply(obj, 1);

        hpx::apply(hpx::util::bind(obj, 1));
        hpx::apply(hpx::util::bind(obj, _1), 1);
    }

#   if !defined(BOOST_NO_CXX11_LAMBDAS) || !defined( BOOST_NO_CXX11_AUTO_DECLARATIONS)
    {
        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

// Lambdas are not assignable...
//
//        hpx::apply(increment_lambda, 1);
//
//        hpx::apply(hpx::util::bind(increment_lambda, 1));
//        hpx::apply(hpx::util::bind(increment_lambda, _1), 1);
    }
#   endif /*!defined(BOOST_NO_CXX11_LAMBDAS) || !defined( BOOST_NO_CXX11_AUTO_DECLARATIONS)*/

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    accumulator.store(0);

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

#   ifndef BOOST_NO_CXX11_LAMBDAS
    HPX_TEST_EQ(accumulator.load(), 12);
#   else /*BOOST_NO_CXX11_LAMBDAS*/
    HPX_TEST_EQ(accumulator.load(), 12);
#   endif /*BOOST_NO_CXX11_LAMBDAS*/

    return hpx::util::report_errors();
}

