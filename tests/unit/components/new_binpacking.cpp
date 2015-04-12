//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::simple_component_base<test_server>
{
    hpx::id_type call() const { return hpx::find_here(); }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION(call_action);

struct test_client : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server> base_type;

    test_client(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    hpx::id_type  call()
    {
        return hpx::async<call_action>(this->get_gid()).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
void test_binpacking()
{
    // create an increasing number of instances on all available localities
    std::vector<std::vector<hpx::id_type> > targets;

    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    for (std::size_t i = 0; i != localities.size(); ++i)
    {
        hpx::id_type const& loc = localities[i];

        targets.push_back(hpx::new_<test_server[]>(loc, i+1).get());
        for (hpx::id_type const& id: targets.back())
        {
            HPX_TEST(hpx::async<call_action>(id).get() == loc);
        }
    }

    std::string counter_name(hpx::components::default_binpacking_counter_name);
    counter_name += "test_server";

    boost::uint64_t count = 0;
    for (std::size_t i = 0; i != localities.size(); ++i)
    {
        hpx::performance_counters::performance_counter instances(
            counter_name, localities[i]);

        count += instances.get_value_sync<boost::uint64_t>();
    }

    // now use bin-packing policy to fill up the number of instances
    std::vector<hpx::id_type> filled_targets =
        hpx::new_<test_server[]>(hpx::binpacked(localities), count).get();

    // now, all localities should have the same number of instances
    boost::uint64_t new_count = 0;
    for (std::size_t i = 0; i != localities.size(); ++i)
    {
        hpx::performance_counters::performance_counter instances(
            counter_name, localities[i]);

        boost::uint64_t c = instances.get_value_sync<boost::uint64_t>();
        new_count += c;

        HPX_TEST_EQ(c, localities.size()+1);
    }

    HPX_TEST_EQ(2*count, new_count);
}

int main()
{
    test_binpacking();
    return 0;
}

