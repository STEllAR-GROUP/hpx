//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_CREATE_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_CREATE_COMPONENT_JUN_02_2008_0146PM

#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/server/create_component_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/functional/new.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unique_function.hpp>

#include <sstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// Create arrays of components using their default constructor
    template <typename Component>
    naming::gid_type create(std::size_t count)
    {
        Component* c = static_cast<Component*>(Component::create(count));

        naming::gid_type gid = c->get_base_gid();
        if (gid)
        {
            // everything is ok, return the new id
            return gid;
        }

        Component::destroy(c, count);

        std::ostringstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "create<Component>",
            strm.str());

        return naming::invalid_gid;
    }

    template <typename Component>
    naming::gid_type create(
        util::unique_function_nonser<void(void*)> const& ctor)
    {
        void * cv = Component::heap_type::alloc(1);
        try {
            ctor(cv);
        }
        catch(...)
        {
            Component::heap_type::free(cv, 1); //-V107
            throw;
        }
        Component *c = static_cast<Component *>(cv);

        naming::gid_type gid = c->get_base_gid();
        if (gid)
        {
            // everything is ok, return the new id
            return gid;
        }

        Component::heap_type::free(c, 1); //-V107

        std::ostringstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "create<Component>(ctor)",
            strm.str());

        return naming::invalid_gid;
    }

    template <typename Component>
    naming::gid_type create(naming::gid_type const& gid,
        util::unique_function_nonser<void(void*)> const& ctor)
    {
        void * cv = Component::heap_type::alloc(1);
        try {
            ctor(cv);
        }
        catch(...)
        {
            Component::heap_type::free(cv, 1); //-V107
            throw;
        }
        Component *c = static_cast<Component *>(cv);

        naming::gid_type assigned_gid = c->get_base_gid(gid);
        if (assigned_gid && assigned_gid == gid)
        {
            // everything is ok, return the new id
            return gid;
        }

        Component::heap_type::free(c, 1); //-V107

        std::ostringstream strm;
        strm << "global id " << assigned_gid <<
            " is already bound to a different component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "create<Component>(naming::gid_type, ctor)",
            strm.str());

        return naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Create component with arguments
    namespace detail
    {
        template <typename Component, typename ...Ts>
        util::detail::bound<
            util::detail::one_shot_wrapper<
                util::functional::placement_new<typename Component::derived_type>
            >(util::detail::placeholder<1> const&, Ts&&...)
        > construct_function(Ts&&... vs)
        {
            typedef typename Component::derived_type type;

            return util::bind(
                util::one_shot(util::functional::placement_new<type>()),
                util::placeholders::_1, std::forward<Ts>(vs)...);
        }
    }

    template <typename Component, typename ...Ts>
    naming::gid_type construct(Ts&&... vs)
    {
        return server::create<Component>(
            detail::construct_function<Component>(std::forward<Ts>(vs)...));
    }
}}}

#endif

