//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::int32_t sequence(boost::int32_t i)
{
    return i+1;
}

HPX_PLAIN_ACTION(sequence);

///////////////////////////////////////////////////////////////////////////////
struct continuation
{
    void operator()(hpx::id_type next, boost::int32_t i) const
    {
        hpx::lcos::base_lco_with_value<boost::int32_t>::set_value_action set_int;
        hpx::apply(set_int, next, boost::move(i));
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    sequence_action seq;

    // test locally
    { 
        hpx::future<int> f = hpx::async_continue(seq, hpx::find_here(), 42, continuation());
        HPX_TEST_EQ(f.get(), 43);
    }

    // test remotely, if possible
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    if (!localities.empty()) 
    {
        hpx::future<int> f = hpx::async_continue(seq, localities[0], 42, continuation());
        HPX_TEST_EQ(f.get(), 43);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

