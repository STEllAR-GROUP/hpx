//  Copyright (c) 2015 Konstantin Kronfeldner
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/format.hpp>

using namespace std;

namespace server
{

struct ViewRegistrationListener
    : hpx::components::simple_component_base<ViewRegistrationListener>
{
    ViewRegistrationListener() :
        name("<unknown>")
    {
        cout << "constructed server listener without name" << endl;
    }

    ViewRegistrationListener(const string &name) :
        name(name)
    {
        cout << boost::format("constructed server listener %1% (%2%)")
            % name % this << endl;
    }

    void register_view()
    {
        cout << boost::format("register view at listener %1% (%2%)")
            % name % this << endl;
    }
    HPX_DEFINE_COMPONENT_ACTION(ViewRegistrationListener, register_view);

    string name;
};

}

typedef hpx::components::simple_component<
        server::ViewRegistrationListener
    > view_registration_listener_type;

HPX_REGISTER_COMPONENT(
    view_registration_listener_type, ViewRegistrationListener);

HPX_REGISTER_ACTION_DECLARATION(
    server::ViewRegistrationListener::register_view_action,
    view_registration_listener_register_view_action);
HPX_REGISTER_ACTION(
    server::ViewRegistrationListener::register_view_action,
    view_registration_listener_register_view_action);

namespace client
{

struct ViewRegistrationListener
  : hpx::components::client_base<
        ViewRegistrationListener, server::ViewRegistrationListener>
{
    typedef hpx::components::client_base<
            ViewRegistrationListener, server::ViewRegistrationListener
        > base_type;

    ViewRegistrationListener(hpx::future<hpx::naming::id_type> gid)
        : base_type(move(gid))
    {
        cout << "constructed listener client by future" << endl;
    }

    ViewRegistrationListener(hpx::naming::id_type gid)
        : base_type(gid)
    {
        cout << "constructed listener client by gid" << endl;
    }

    void register_view()
    {
        typedef server::ViewRegistrationListener::register_view_action
            action_type;
        hpx::async<action_type>(this->get_id()).get();
    }
};

}

int hpx_main()
{
    std::size_t num_expected_ids = 0;
    {
        auto id = hpx::new_<server::ViewRegistrationListener>(
            hpx::find_here(), string("A")).get();
        bool result = hpx::register_with_basename("Listener", id).get();
        HPX_TEST(result);
        if (result)
            ++num_expected_ids;
    }

    {
        auto id = hpx::new_<server::ViewRegistrationListener>(
            hpx::find_here(), string("B")).get();
        bool result = hpx::register_with_basename("Listener", id).get();
        HPX_TEST(!result);
        if (result)
            ++num_expected_ids;
    }

    {
        auto id = hpx::new_<server::ViewRegistrationListener>(
            hpx::find_here(), string("C")).get();
        bool result = hpx::register_with_basename("Listener", id).get();
        HPX_TEST(!result);
        if (result)
            ++num_expected_ids;
    }

    {
        auto ids = hpx::find_all_from_basename("Listener", num_expected_ids);

        for (auto &f : ids)
        {
            cout << "trying to get id" << endl;

            auto id = f.get(); // works for the first element, hangs for the others

            cout << "resolved id: " << id << endl;

            client::ViewRegistrationListener client(id);
            cout << "created client with id " << client.get_id() << endl;

            client.register_view();
            client.register_view();

            cout << "registered everything at this id" << endl;
        }
    }

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
