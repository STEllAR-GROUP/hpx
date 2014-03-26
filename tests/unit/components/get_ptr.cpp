//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::simple_component_base<test_server>
{
    std::size_t check_ptr() const
    {
        return reinterpret_cast<std::size_t>(this);
    }

    HPX_DEFINE_COMPONENT_CONST_ACTION(test_server, check_ptr, check_ptr_action);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_type, test_server);

typedef test_server::check_ptr_action check_ptr_action;
HPX_REGISTER_ACTION_DECLARATION(check_ptr_action);
HPX_REGISTER_ACTION(check_ptr_action);

struct test_client 
  : hpx::components::client_base<test_client, test_server>
{
    std::size_t check_ptr() { return check_ptr_action()(this->get_gid()); }
};

///////////////////////////////////////////////////////////////////////////////
bool test_get_ptr1(hpx::id_type id)
{
    test_client t;
    t.create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_gid());

    try {
        hpx::future<boost::shared_ptr<test_server> > f =
            hpx::get_ptr<test_server>(t.get_gid());

        boost::shared_ptr<test_server> ptr = f.get();

        HPX_TEST_EQ(reinterpret_cast<test_server*>(t.check_ptr()), ptr.get());
        return true;
    }
    catch (hpx::exception const& e) {
        HPX_TEST_EQ(e.get_error(), hpx::bad_parameter);
    }

    return false;
}

bool test_get_ptr2(hpx::id_type id)
{
    test_client t;
    t.create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_gid());

    hpx::future<boost::shared_ptr<test_server> > f =
        hpx::get_ptr<test_server>(t.get_gid());

    hpx::error_code ec;
    boost::shared_ptr<test_server> ptr = f.get(ec);
    if (ec) return false;

    HPX_TEST_EQ(reinterpret_cast<test_server*>(t.check_ptr()), ptr.get());
    return true;
}

int main()
{
    HPX_TEST(test_get_ptr1(hpx::find_here()));
    HPX_TEST(test_get_ptr2(hpx::find_here()));

    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    BOOST_FOREACH(hpx::id_type const& id, localities)
    {
        HPX_TEST(!test_get_ptr1(id));
        HPX_TEST(!test_get_ptr2(id));
    }

    return hpx::util::report_errors();
}

