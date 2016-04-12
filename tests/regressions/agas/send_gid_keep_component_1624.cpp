//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace server
{
    struct view_registry :
        hpx::components::managed_component_base<view_registry>
    {
        int register_view(const hpx::naming::id_type &gid)
        {
            std::cout <<
                boost::format("register view at %1% by %2%") %
                    this->get_unmanaged_id() % gid << std::endl;
            registered_regions_.push_back(gid);
            return 0;
        }

        void update();

        HPX_DEFINE_COMPONENT_ACTION(view_registry, register_view);
        HPX_DEFINE_COMPONENT_ACTION(view_registry, update);

        std::vector<hpx::naming::id_type> registered_regions_;
    };
}

HPX_REGISTER_ACTION_DECLARATION(server::view_registry::register_view_action,
    view_registration_listener_register_view_action);
HPX_REGISTER_ACTION_DECLARATION(server::view_registry::update_action,
    view_registration_listener_update_action);


namespace client
{
    struct view_registry
    {
        view_registry(hpx::naming::id_type gid) :
            gid_(gid)
        {
        }

        void register_view(const hpx::naming::id_type &viewerId)
        {
            hpx::async<server::view_registry::register_view_action>(
                gid_, viewerId).get();
        }

        void update()
        {
            hpx::async<server::view_registry::update_action>(gid_).get();
        }

        hpx::naming::id_type gid_;
    };
}

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<server::view_registry>
    view_registration_listener_type;
HPX_REGISTER_COMPONENT(
    view_registration_listener_type, view_registry);

HPX_REGISTER_ACTION(server::view_registry::register_view_action,
    view_registration_listener_register_view_action);
HPX_REGISTER_ACTION(server::view_registry::update_action,
    view_registration_listener_update_action);


///////////////////////////////////////////////////////////////////////////////
namespace server
{
    struct viewer : hpx::components::managed_component_base<viewer>
    {
        ~viewer()
        {
            std::cout << "~viewer" << std::endl;
        }

        // This function does a broadcast to all writers to register a certain
        // region
        void register_region(const std::vector<hpx::naming::id_type> &writers)
        {
            for (const auto &id : writers)
            {
                std::cout <<
                    boost::format("%1% registering region at %2%") %
                        this->get_unmanaged_id() % id << std::endl;

                client::view_registry listener(id);
                listener.register_view(this->get_id());
            }
        }

        void update()
        {
            HPX_TEST(this->get_unmanaged_id());
            std::cout << "update() at viewer "
                << this->get_unmanaged_id() << std::endl;
        }

        HPX_DEFINE_COMPONENT_ACTION(viewer, register_region);
        HPX_DEFINE_COMPONENT_ACTION(viewer, update);
    };
}

HPX_REGISTER_ACTION_DECLARATION(server::viewer::register_region_action,
    viewer_register_region_action);
HPX_REGISTER_ACTION_DECLARATION(server::viewer::update_action,
    viewer_update_action);

namespace client
{
    struct viewer
    {
        viewer(const hpx::naming::id_type &gid) :
            gid_(gid)
        {
        }

        void register_region(const std::vector<hpx::naming::id_type> &writers)
        {
            hpx::async<server::viewer::register_region_action>(
                gid_, writers).get();
        }

        void update()
        {
            hpx::async<server::viewer::update_action>(gid_);
        }

        hpx::naming::id_type gid_;
    };
}

typedef hpx::components::managed_component<server::viewer> viewer_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(viewer_type, viewer);

HPX_REGISTER_ACTION(server::viewer::update_action, viewer_update_action);
HPX_REGISTER_ACTION(server::viewer::register_region_action,
    viewer_register_region_action);


///////////////////////////////////////////////////////////////////////////////
namespace server
{
    void view_registry::update()
    {
        for (auto &id : registered_regions_)
        {
            std::cout << " sending update data to " << id << std::endl;
            client::viewer viewer(id);
            viewer.update();
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        auto registryId = hpx::new_<server::view_registry>(hpx::find_here()).get();
        client::view_registry registry(registryId);

        // create a viewer
        {
            auto id = hpx::new_<server::viewer>(hpx::find_here()).get();
            client::viewer viewer(id);
            std::vector<hpx::id_type> ids;
            ids.push_back(registryId);
            viewer.register_region(ids);

            // viewer's gid is saved in the registry's vector
            registry.update();

            std::cout << "viewer " << id << " will go out of scope" << std::endl;
        }

        std::cout << "viewer went out of scope" << std::endl;

        // calling update() on the viewer will fail here
        for (int i = 0; i < 15; ++i)
        {
            registry.update();
        }
    }

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

