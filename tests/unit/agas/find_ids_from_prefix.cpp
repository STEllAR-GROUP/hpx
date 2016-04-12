//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::simple_component_base<test_server>
{
    test_server()
    {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION(call_action);

struct test_client
  : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server>
        base_type;

    test_client() {}

    template <typename ... Ts>
    test_client(Ts && ...vs)
      : base_type(std::forward<Ts>(vs)...)
    {}

    hpx::id_type call() const { return call_action()(this->get_id()); }
};

///////////////////////////////////////////////////////////////////////////////
void test_find_all_ids_from_basename()
{
    char const* basename = "/find_all_ids_from_prefix_test/";

    test_client t1 = test_client::create(hpx::find_here());
    hpx::id_type client_id = t1.get_id();

    HPX_TEST_NEQ(hpx::naming::invalid_id, client_id);

    // register our component with AGAS
    HPX_TEST((hpx::register_with_basename(basename, client_id).get()));

    // wait for all localities to register their component
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    std::vector<hpx::future<hpx::id_type> > all_ids =
        hpx::find_all_from_basename(basename, localities.size());
    HPX_TEST_EQ(all_ids.size(), localities.size());

    // retrieve all component ids
    std::set<hpx::id_type> component_localities;
    for (hpx::future<hpx::id_type>& f : all_ids)
    {
        hpx::id_type id = f.get();
        hpx::id_type locality = test_client(id).call();
        std::pair<std::set<hpx::id_type>::iterator, bool> p =
            component_localities.insert(locality);

        HPX_TEST(p.second);     // every id should be unique
    }
    HPX_TEST_EQ(component_localities.size(), localities.size());

    // make sure that components are on all localities
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(component_localities.find(id) != component_localities.end());
    }
}

void test_find_ids_from_basename()
{
    char const* basename = "/find_ids_from_prefix_test/";

    test_client t1 = test_client::create(hpx::find_here());
    hpx::id_type client_id = t1.get_id();

    HPX_TEST_NEQ(hpx::naming::invalid_id, client_id);

    // register our component with AGAS
    HPX_TEST((hpx::register_with_basename(basename, client_id).get()));

    // wait for all localities to register their component
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    std::vector<std::size_t> sequence_nrs;
    sequence_nrs.reserve(localities.size());
    for (hpx::id_type const& locality : localities)
    {
        sequence_nrs.push_back(hpx::naming::get_locality_id_from_id(locality));
    }

    std::vector<hpx::future<hpx::id_type> > ids =
        hpx::find_from_basename(basename, sequence_nrs);
    HPX_TEST_EQ(ids.size(), sequence_nrs.size());

    // retrieve all component ids
    std::set<hpx::id_type> component_localities;
    for (hpx::future<hpx::id_type>& f : ids)
    {
        hpx::id_type id = f.get();

        hpx::id_type locality = test_client(id).call();
        std::pair<std::set<hpx::id_type>::iterator, bool> p =
            component_localities.insert(locality);

        HPX_TEST(p.second);     // every id should be unique
    }
    HPX_TEST_EQ(component_localities.size(), localities.size());

    // make sure that components are on all localities
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(component_localities.find(id) != component_localities.end());
    }
}

void test_find_id_from_basename()
{
    char const* basename = "/find_id_from_prefix_test/";

    test_client t1 = test_client::create(hpx::find_here());
    hpx::id_type client_id = t1.get_id();

    HPX_TEST_NEQ(hpx::naming::invalid_id, client_id);

    // register our component with AGAS
    HPX_TEST((hpx::register_with_basename(basename, client_id).get()));

    // wait for all localities to register their component
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    std::vector<std::size_t> sequence_nrs;
    std::vector<hpx::future<hpx::id_type> > ids;
    sequence_nrs.reserve(localities.size());
    ids.reserve(localities.size());

    for (hpx::id_type const& locality : localities)
    {
        std::size_t nr = hpx::naming::get_locality_id_from_id(locality);
        sequence_nrs.push_back(nr);
        ids.push_back(hpx::find_from_basename(basename, nr));
    }

    HPX_TEST_EQ(ids.size(), sequence_nrs.size());

    // retrieve all component ids
    std::set<hpx::id_type> component_localities;
    for (hpx::future<hpx::id_type>& f : ids)
    {
        hpx::id_type id = f.get();

        hpx::id_type locality = test_client(id).call();
        std::pair<std::set<hpx::id_type>::iterator, bool> p =
            component_localities.insert(locality);

        HPX_TEST(p.second);     // every id should be unique
    }
    HPX_TEST_EQ(component_localities.size(), localities.size());

    // make sure that components are on all localities
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(component_localities.find(id) != component_localities.end());
    }
}

int hpx_main()
{
    test_find_id_from_basename();
    test_find_ids_from_basename();
    test_find_all_ids_from_basename();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

