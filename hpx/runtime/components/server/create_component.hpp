//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_CREATE_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_CREATE_COMPONENT_JUN_02_2008_0146PM

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/server/create_component_fwd.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/functional/new.hpp>

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
    naming::gid_type create(util::function_nonser<void(void*)> const& ctor)
    {
        Component* c = (Component*)Component::heap_type::alloc(1);
        ctor(c);

        naming::gid_type gid = c->get_base_gid();
        if (gid)
        {
            // everything is ok, return the new id
            return gid;
        }

        Component::heap_type::free(c, 1);

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
        util::function_nonser<void(void*)> const& ctor)
    {
        Component* c = (Component*)Component::heap_type::alloc(1);
        ctor(c);

        naming::gid_type assigned_gid = c->get_base_gid(gid);
        if (assigned_gid && assigned_gid == gid)
        {
            // everything is ok, return the new id
            return gid;
        }

        Component::heap_type::free(c, 1);

        std::ostringstream strm;
        strm << "global id " << assigned_gid <<
            " is already bound to a different component instance";
        HPX_THROW_EXCEPTION(hpx::duplicate_component_address,
            "create<Component>(naming::gid_type, ctor)",
            strm.str());

        return naming::invalid_gid;
    }

    template <typename Component>
    Component* internal_create(typename Component::wrapped_type* impl)
    {
        Component* p = Component::heap_type::alloc(1);
        return new (p) typename Component::derived_type(impl);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Create component with arguments
    namespace detail
    {
        template <typename Component, typename ...Ts>
        util::detail::bound<
            util::detail::one_shot_wrapper<
                util::functional::placement_new<typename Component::derived_type>
            >,
            util::tuple<
                util::detail::placeholder<1>,
                typename util::decay<Ts>::type...
            >
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

