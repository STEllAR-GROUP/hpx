//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described by #1813:
// async(launch::..., action(), ...) always invokes locally

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
hpx::id_type get_locality()
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(get_locality);

///////////////////////////////////////////////////////////////////////////////
struct get_locality_server
  : hpx::components::simple_component_base<get_locality_server>
{
    hpx::id_type call()
    {
        return hpx::find_here();
    }

    HPX_DEFINE_COMPONENT_ACTION(get_locality_server, call);
};

typedef hpx::components::simple_component<get_locality_server> server_type;
HPX_REGISTER_COMPONENT(server_type, get_locality_server);

typedef get_locality_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
void test_remote_async(hpx::id_type target)
{
    {
        get_locality_action act;

        hpx::future<hpx::id_type> f1 = hpx::async(act, target);
        HPX_TEST_EQ(f1.get(), target);

        hpx::future<hpx::id_type> f2 =
            hpx::async(hpx::launch::all, act, target);
        HPX_TEST_EQ(f2.get(), target);
    }

    {
        hpx::future<hpx::id_type> obj_f =
            hpx::components::new_<get_locality_server>(target);
        hpx::id_type obj = obj_f.get();

        call_action call;

        hpx::future<hpx::id_type> f1 = hpx::async(call, obj);
        HPX_TEST_EQ(f1.get(), target);

        hpx::future<hpx::id_type> f2 =
            hpx::async(hpx::launch::all, call, obj);
        HPX_TEST_EQ(f2.get(), target);
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

