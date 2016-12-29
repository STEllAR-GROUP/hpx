//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::simple_component_base<test_server>
{
    std::size_t check_ptr() const
    {
        return reinterpret_cast<std::size_t>(this);
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, check_ptr, check_ptr_action);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::check_ptr_action check_ptr_action;
HPX_REGISTER_ACTION_DECLARATION(check_ptr_action);
HPX_REGISTER_ACTION(check_ptr_action);

struct test_client
  : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server> base_type;

    test_client(hpx::future<hpx::id_type>&& id) : base_type(std::move(id)) {}

    std::size_t check_ptr() { return check_ptr_action()(this->get_id()); }
};

///////////////////////////////////////////////////////////////////////////////
bool test_get_ptr1(hpx::id_type id)
{
    test_client t = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_id());

    try {
        hpx::future<std::shared_ptr<test_server> > f =
            hpx::get_ptr<test_server>(t.get_id());

        std::shared_ptr<test_server> ptr = f.get();

        HPX_TEST_EQ(reinterpret_cast<test_server*>(t.check_ptr()), ptr.get());
        return true;
    }
    catch (hpx::exception const& e) {
        HPX_TEST_EQ(int(e.get_error()), int(hpx::bad_parameter));
    }

    return false;
}

bool test_get_ptr2(hpx::id_type id)
{
    test_client t = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_id());

    hpx::future<std::shared_ptr<test_server> > f =
        hpx::get_ptr<test_server>(t.get_id());

    f.wait();
    bool has_exception = f.has_exception();

    hpx::error_code ec;
    std::shared_ptr<test_server> ptr = f.get(ec);

    // Intel 13 has trouble to generate correct code for if(ec) { ... }
    if (ec || !ptr.get())
    {
        HPX_TEST(has_exception);
        return false;
    }

    HPX_TEST(!has_exception);
    HPX_TEST_EQ(reinterpret_cast<test_server*>(t.check_ptr()), ptr.get());
    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool test_get_ptr3(hpx::id_type id)
{
    test_client t = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_id());

    try {
        hpx::future<std::shared_ptr<test_server> > f = hpx::get_ptr(t);
        std::shared_ptr<test_server> ptr = f.get();

        HPX_TEST_EQ(reinterpret_cast<test_server*>(t.check_ptr()), ptr.get());
        return true;
    }
    catch (hpx::exception const& e) {
        HPX_TEST_EQ(int(e.get_error()), int(hpx::bad_parameter));
    }

    return false;
}

bool test_get_ptr4(hpx::id_type id)
{
    test_client t = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_id());

    hpx::future<std::shared_ptr<test_server> > f = hpx::get_ptr(t);
    f.wait();

    bool has_exception = f.has_exception();

    hpx::error_code ec;
    std::shared_ptr<test_server> ptr = f.get(ec);

    // Intel 13 has trouble to generate correct code for if(ec) { ... }
    if (ec || !ptr.get())
    {
        HPX_TEST(has_exception);
        return false;
    }

    HPX_TEST(!has_exception);
    HPX_TEST_EQ(reinterpret_cast<test_server*>(t.check_ptr()), ptr.get());
    return true;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    HPX_TEST(test_get_ptr1(hpx::find_here()));
    HPX_TEST(test_get_ptr2(hpx::find_here()));
    HPX_TEST(test_get_ptr3(hpx::find_here()));
    HPX_TEST(test_get_ptr4(hpx::find_here()));


    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(!hpx::expect_exception());

        HPX_TEST(!test_get_ptr1(id));
        HPX_TEST(!test_get_ptr2(id));
        HPX_TEST(!test_get_ptr3(id));
        HPX_TEST(!test_get_ptr4(id));

        HPX_TEST(hpx::expect_exception(false));
    }

    return hpx::util::report_errors();
}

