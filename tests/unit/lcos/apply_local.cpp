//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <chrono>
#include <functional>
#include <mutex>

///////////////////////////////////////////////////////////////////////////////
boost::atomic<boost::int32_t> accumulator;
hpx::lcos::local::condition_variable_any result_cv;

void increment(boost::int32_t i)
{
    accumulator += i;
    result_cv.notify_one();
}

void increment_with_future(hpx::shared_future<boost::int32_t> fi)
{
    accumulator += fi.get();
    result_cv.notify_one();
}

///////////////////////////////////////////////////////////////////////////////
struct increment_function_object
{
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

#if defined(HPX_HAVE_CXX11_LAMBDAS) && defined(HPX_HAVE_CXX11_AUTO)
auto increment_lambda = [](boost::int32_t i){ accumulator += i; };
#endif

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
        hpx::promise<boost::int32_t> p;
        hpx::shared_future<boost::int32_t> f = p.get_future();

        using hpx::util::placeholders::_1;

        hpx::apply(&increment_with_future, f);
        hpx::apply(hpx::util::bind(&increment_with_future, f));
        hpx::apply(hpx::util::bind(&increment_with_future, _1), f);

        p.set_value(1);
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

#   if defined(HPX_HAVE_CXX11_LAMBDAS) && defined(HPX_HAVE_CXX11_AUTO)
    {
        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        hpx::apply(increment_lambda, 1);
        hpx::apply(hpx::util::bind(increment_lambda, 1));
        hpx::apply(hpx::util::bind(increment_lambda, _1), 1);
    }
#   endif

    hpx::lcos::local::no_mutex result_mutex;
    std::unique_lock<hpx::lcos::local::no_mutex> l(result_mutex);
#   if defined(HPX_HAVE_CXX11_LAMBDAS) && defined(HPX_HAVE_CXX11_AUTO)
    result_cv.wait_for(l, std::chrono::seconds(1),
        hpx::util::bind(std::equal_to<boost::int32_t>(), std::ref(accumulator), 18));

    HPX_TEST_EQ(accumulator.load(), 18);
#   else
    result_cv.wait_for(l, std::chrono::seconds(1),
        hpx::util::bind(std::equal_to<boost::int32_t>(), std::ref(accumulator), 15));

    HPX_TEST_EQ(accumulator.load(), 15);
#   endif

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    accumulator.store(0);

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

