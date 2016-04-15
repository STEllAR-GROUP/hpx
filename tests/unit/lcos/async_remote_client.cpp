//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

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
    typedef hpx::components::client<decrement_server> decrement_client;

    {
        decrement_client dec_f =
            hpx::components::new_<decrement_client>(target);

        call_action call;
        hpx::future<boost::int32_t> f1 = hpx::async(call, dec_f, 42);
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::launch::all, call, dec_f, 42);
        HPX_TEST_EQ(f2.get(), 41);
    }

    {
        decrement_client dec_f =
            hpx::components::new_<decrement_client>(target);

        call_action call;

        hpx::future<boost::int32_t> f1 =
            hpx::async(hpx::util::bind(call, dec_f, 42));
        HPX_TEST_EQ(f1.get(), 41);

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        hpx::future<boost::int32_t> f2 =
            hpx::async(hpx::util::bind(call, _1, 42), dec_f);
        HPX_TEST_EQ(f2.get(), 41);

        hpx::future<boost::int32_t> f3 =
            hpx::async(hpx::util::bind(call, _1, _2), dec_f, 42);
        HPX_TEST_EQ(f3.get(), 41);
    }

    {
        decrement_client dec_f =
            hpx::components::new_<decrement_client>(target);

        hpx::future<boost::int32_t> f1 =
            hpx::async<call_action>(dec_f, 42);
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<boost::int32_t> f2 =
            hpx::async<call_action>(hpx::launch::all, dec_f, 42);
        HPX_TEST_EQ(f2.get(), 41);
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

