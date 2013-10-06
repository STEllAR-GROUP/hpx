//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_CREATE_COMPONENT_JUN_02_2008_0146PM)
#define HPX_COMPONENTS_SERVER_CREATE_COMPONENT_JUN_02_2008_0146PM

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/stringstream.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// Create arrays of components using their default constructor
    template <typename Component>
    naming::gid_type create(std::size_t count = 1, error_code& ec = throws)
    {
        Component* c = static_cast<Component*>(Component::create(count));
        naming::gid_type gid = c->get_base_gid();

        if (gid)
        {
            // NOTE: This code is unnecessary, the address is already in the
            // cache, because bind_range/bind caches it. 
            //naming::resolver_client& cl = naming::get_agas_client();
            //cl.update_cache(gid, cl.get_here(),
            //    components::get_component_type<typename Component::wrapped_type>(),
            //    reinterpret_cast<boost::uint64_t>(c), count, ec);

            // everything is ok, return the new id
            if (&ec != &throws)
                ec = make_success_code();
            return gid;
        }

        Component::destroy(c, count);

        hpx::util::osstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROWS_IF(ec, hpx::duplicate_component_address,
            "create<Component>",
            hpx::util::osstream_get_string(strm));

        return naming::invalid_gid;
    }

    template <typename Component>
    naming::gid_type create(HPX_STD_FUNCTION<void(void*)> const& ctor,
        error_code& ec = throws)
    {
        Component* c = Component::heap_type::alloc(1);
        ctor(c);

        naming::gid_type gid = c->get_base_gid();

        if (gid)
        {
            // NOTE: This code is unnecessary, the address is already in the
            // cache, because bind_range/bind caches it. 
            //naming::resolver_client& cl = naming::get_agas_client();
            //cl.update_cache(gid, cl.get_here(),
            //    components::get_component_type<typename Component::wrapped_type>(),
            //    reinterpret_cast<boost::uint64_t>(c), 1, ec);

            // everything is ok, return the new id
            if (&ec != &throws)
                ec = make_success_code();
            return gid;
        }

        Component::heap_type::free(c, 1);

        hpx::util::osstream strm;
        strm << "global id " << gid << " is already bound to a different "
                "component instance";
        HPX_THROWS_IF(ec, hpx::duplicate_component_address,
            "create<Component>(ctor)",
            hpx::util::osstream_get_string(strm));

        return naming::invalid_gid;
    }
}}}

#endif

