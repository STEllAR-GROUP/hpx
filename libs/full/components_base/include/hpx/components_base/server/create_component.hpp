//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2011-2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/server/component_heap.hpp>
#include <hpx/components_base/server/create_component_fwd.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/address.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server {

    ///////////////////////////////////////////////////////////////////////////
    /// Create a component and forward the passed parameters
    template <typename Component, typename... Ts>
    naming::gid_type create(Ts&&... ts)
    {
        component_type type =
            get_component_type<typename Component::wrapped_type>();
        if (!enabled(type))
        {
            HPX_THROW_EXCEPTION(bad_request, "components::server::::create",
                "the component is disabled for this locality ({})",
                get_component_type_name(type));
            return naming::invalid_gid;
        }

        void* storage = component_heap<Component>().alloc(1);

        Component* c = nullptr;
        try
        {
            c = new (storage) Component(HPX_FORWARD(Ts, ts)...);
        }
        catch (...)
        {
            component_heap<Component>().free(c, 1);
            throw;
        }

        naming::gid_type gid = c->get_base_gid();
        if (!gid)
        {
            c->finalize();
            c->~Component();
            component_heap<Component>().free(c, 1);

            HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                "create<Component>", "can't assign global id");
        }
        ++instance_count(type);

        return gid;
    }

    template <typename Component, typename... Ts>
    naming::gid_type create_migrated(
        naming::gid_type const& gid, void** p, Ts&&... ts)
    {
        component_type type =
            get_component_type<typename Component::wrapped_type>();
        if (!enabled(type))
        {
            HPX_THROW_EXCEPTION(bad_request,
                "components::server::create_migrated",
                "the component is disabled for this locality ({})",
                get_component_type_name(type));
            return naming::invalid_gid;
        }

        void* storage = component_heap<Component>().alloc(1);

        Component* c = nullptr;
        try
        {
            c = new (storage) Component(HPX_FORWARD(Ts, ts)...);
        }
        catch (...)
        {
            component_heap<Component>().free(c, 1);
            throw;
        }

        naming::gid_type assigned_gid = c->get_base_gid(gid);
        if (assigned_gid && assigned_gid == gid)
        {
            // everything is ok, return the new id
            if (p != nullptr)
            {
                *p = c;    // return the raw address as well
            }
            ++instance_count(type);
            return gid;
        }

        c->finalize();
        c->~Component();
        component_heap<Component>().free(c, 1);

        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "create<Component>(naming::gid_type, ctor)",
            "the global id {} is already bound to a different component "
            "instance",
            gid);

        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Create count components and forward the passed parameters
    template <typename Component, typename... Ts>
    std::vector<naming::gid_type> bulk_create(std::size_t count, Ts&&... ts)
    {
        component_type type =
            get_component_type<typename Component::wrapped_type>();
        std::vector<naming::gid_type> gids;
        if (!enabled(type))
        {
            HPX_THROW_EXCEPTION(bad_request, "components::server::bulk_create",
                "the component is disabled for this locality ({})",
                get_component_type_name(type));
            return gids;
        }

        gids.reserve(count);

        Component* storage =
            static_cast<Component*>(component_heap<Component>().alloc(count));
        Component* storage_it = storage;
        std::size_t succeeded = 0;
        try
        {
            // Call constructors and try to get the GID...
            for (std::size_t i = 0; i != count; ++i, ++storage_it)
            {
                Component* c = nullptr;
                c = new (storage_it) Component(HPX_FORWARD(Ts, ts)...);
                naming::gid_type gid = c->get_base_gid();
                if (!gid)
                {
                    c->finalize();
                    c->~Component();
                    HPX_THROW_EXCEPTION(hpx::unknown_component_address,
                        "bulk_create<Component>", "can't assign global id");
                }
                gids.push_back(HPX_MOVE(gid));
                ++instance_count(type);
                ++succeeded;
            }
        }
        catch (...)
        {
            // If an exception was thrown, roll back
            storage_it = storage;
            for (std::size_t i = 0; i != succeeded; ++i, ++storage_it)
            {
                storage_it->finalize();
                storage_it->~Component();
                --instance_count(type);
            }
            component_heap<Component>().free(storage, count);
            throw;
        }

        return gids;
    }
}}}    // namespace hpx::components::server
