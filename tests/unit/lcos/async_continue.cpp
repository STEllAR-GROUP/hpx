//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::int32_t increment(boost::int32_t i)
{
    return i + 1;
}
HPX_PLAIN_ACTION(increment);

struct continuation
{
    void operator()(hpx::id_type const& next, boost::int32_t i) const
    {
        hpx::set_lco_value(next, i);
    }
};

///////////////////////////////////////////////////////////////////////////////
boost::int32_t mult2(boost::int32_t i)
{
    return i * 2;
}
HPX_PLAIN_ACTION(mult2);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    increment_action inc;

    // test locally
    {
        hpx::future<int> f = hpx::async_continue(inc, hpx::find_here(), 42, continuation());
        HPX_TEST_EQ(f.get(), 43);
    }

    // test remotely, if possible
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty())
    {
        hpx::future<int> f = hpx::async_continue(inc, localities[0], 42, continuation());
        HPX_TEST_EQ(f.get(), 43);
    }

    // test chaining locally
//    {
//        mult2_action mult;
//        hpx::future<int> f = hpx::async_continue(inc, localities[0], 42, 
//            util::bind(mult, _1, _2));
//
//    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

