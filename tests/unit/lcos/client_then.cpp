//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

////////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::component_base<test_server>
{
    hpx::id_type call() const { return hpx::find_here(); }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

struct test : hpx::components::client_base<test, test_server>
{
    typedef hpx::components::client_base<test, test_server> base_type;

    test(hpx::future<hpx::id_type> && id) : base_type(std::move(id)) {}
    test(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}

    hpx::future<hpx::id_type> call() const
    {
        return hpx::async<call_action>(this->get_id());
    }

    hpx::id_type sync_call() const
    {
        return call().get();
    }
};

////////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        test hw = hpx::new_<test>(hpx::find_here());

        hpx::future<hpx::id_type> f =
            hw.then(
                [](test && t)
                {
                    return t.call();
                });

        HPX_TEST_EQ(f.get(), hpx::find_here());
    }

    {
        test hw = hpx::new_<test>(hpx::find_here());

        hpx::future<hpx::id_type> f =
            hw.then(
                [](test && t)
                {
                    return t.sync_call();
                });

        HPX_TEST_EQ(f.get(), hpx::find_here());
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

