//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    void destroy_base_lco(naming::gid_type const& gid,
        util::one_size_heap_list_base* heap, components::component_type type,
        error_code& ec)
    {
        // retrieve the local address bound to the given global id
        applier::applier& appl = hpx::applier::get_applier();
        naming::address addr;
        if (!appl.get_agas_client().resolve(gid, addr))
        {
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to any "
                    "component instance";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy_base_lco", hpx::util::osstream_get_string(strm));
            return;
        }

        // make sure this component is located here
        if (appl.here() != addr.locality_)
        {
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to any local "
                    "component instance";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy_base_lco", hpx::util::osstream_get_string(strm));
            return;
        }

        // make sure it's the correct component type
        if (!types_are_compatible(type, addr.type_))
        {
            hpx::util::osstream strm;
            strm << "global id " << gid << " is not bound to a component "
                    "instance of type: " << get_component_type_name(type)
                 << " (it is bound to a " << get_component_type_name(addr.type_)
                 << ")";
            HPX_THROWS_IF(ec, hpx::unknown_component_address,
                "destroy_base_lco", hpx::util::osstream_get_string(strm));
            return;
        }

        // we know that this function is used with promises only
        using components::managed_promise;
        managed_promise* promise = reinterpret_cast<managed_promise*>(addr.address_);

        promise->~managed_promise();    // call destructor of derived promise
        heap->free(promise);            // ask heap to free memory

        if (&ec != &throws)
            ec = make_success_code();
    }
}}}

