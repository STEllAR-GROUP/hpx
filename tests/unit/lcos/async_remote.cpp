//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::int32_t increment(boost::int32_t i)
{
    return i + 1;
}
HPX_PLAIN_ACTION(increment);

///////////////////////////////////////////////////////////////////////////////
struct decrement_server
  : hpx::components::managed_component_base<decrement_server>
{
    boost::int32_t call(boost::int32_t i) const
    {
        return i - 1;
    }

    HPX_DEFINE_COMPONENT_CONST_ACTION(decrement_server, call);
};

typedef hpx::components::managed_component<decrement_server> server_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_type, decrement_server);

typedef decrement_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::id_type here = hpx::find_here();

    {
        increment_action inc;

        hpx::future<boost::int32_t> f1 = hpx::async(inc, here, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::launch::all, inc, here, 42);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        increment_action inc;

        hpx::future<boost::int32_t> f1 =
            hpx::async(hpx::util::bind(inc, here, 42));
        HPX_TEST_EQ(f1.get(), 43);
    }

    {
        hpx::future<boost::int32_t> f1 = hpx::async<increment_action>(here, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<boost::int32_t> f2 =
            hpx::async<increment_action>(hpx::launch::all, here, 42);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(here);
        hpx::id_type dec = dec_f.get();

        call_action call;

        hpx::future<boost::int32_t> f1 = hpx::async(call, dec, 42);
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::launch::all, call, dec, 42);
        HPX_TEST_EQ(f2.get(), 41);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(here);
        hpx::id_type dec = dec_f.get();

        call_action call;

        hpx::future<boost::int32_t> f1 =
            hpx::async(hpx::util::bind(call, dec, 42));
        HPX_TEST_EQ(f1.get(), 41);

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::util::bind(call, _1, 42), dec);
        HPX_TEST_EQ(f2.get(), 41);

        hpx::future<boost::int32_t> f3 =
            hpx::async(hpx::util::bind(call, _1, _2), dec, 42);
        HPX_TEST_EQ(f3.get(), 41);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(here);
        hpx::id_type dec = dec_f.get();

        hpx::future<boost::int32_t> f1 = hpx::async<call_action>(dec, 42);
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<boost::int32_t> f2 =
            hpx::async<call_action>(hpx::launch::all, dec, 42);
        HPX_TEST_EQ(f2.get(), 41);
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

