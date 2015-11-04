//  Copyright (c) 2007-2014 Hartmut Kaiser
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

boost::int32_t increment_with_future(hpx::shared_future<boost::int32_t> fi)
{
    return fi.get() + 1;
}
HPX_PLAIN_ACTION(increment_with_future);

///////////////////////////////////////////////////////////////////////////////
struct decrement_server
  : hpx::components::managed_component_base<decrement_server>
{
    boost::int32_t call(boost::int32_t i) const
    {
        return i - 1;
    }

    HPX_DEFINE_COMPONENT_ACTION(decrement_server, call);
};

typedef hpx::components::managed_component<decrement_server> server_type;
HPX_REGISTER_COMPONENT(server_type, decrement_server);

typedef decrement_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
void test_remote_async(hpx::id_type const& target)
{
    {
        increment_action inc;

        hpx::future<boost::int32_t> f1 = hpx::async(inc, target, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::launch::all, inc, target, 42);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        increment_with_future_action inc;
        hpx::promise<boost::int32_t> p;
        hpx::shared_future<boost::int32_t> f = p.get_future();

        hpx::future<boost::int32_t> f1 = hpx::async(inc, target, f);
        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::launch::all, inc, target, f);

        p.set_value(42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        increment_action inc;

        hpx::future<boost::int32_t> f1 =
            hpx::async(hpx::util::bind(inc, target, 42));
        HPX_TEST_EQ(f1.get(), 43);
    }

    {
        hpx::future<boost::int32_t> f1 =
            hpx::async<increment_action>(target, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<boost::int32_t> f2 =
            hpx::async<increment_action>(hpx::launch::all, target, 42);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(target);
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
            hpx::components::new_<decrement_server>(target);
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
            hpx::components::new_<decrement_server>(target);
        hpx::id_type dec = dec_f.get();

        hpx::future<boost::int32_t> f1 =
            hpx::async<call_action>(dec, 42);
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<boost::int32_t> f2 =
            hpx::async<call_action>(hpx::launch::all, dec, 42);
        HPX_TEST_EQ(f2.get(), 41);
    }

    {
        increment_with_future_action inc;
        hpx::shared_future<boost::int32_t> f =
            hpx::async(hpx::launch::deferred, hpx::util::bind(&increment, 42));

        hpx::future<boost::int32_t> f1 = hpx::async(inc, target, f);
        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::launch::all, inc, target, f);

        HPX_TEST_EQ(f1.get(), 44);
        HPX_TEST_EQ(f2.get(), 44);
    }
}

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    for (hpx::id_type const& id : localities)
    {
        test_remote_async(id);
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

